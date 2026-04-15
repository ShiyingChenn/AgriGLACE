import os
import sys
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# ===== 让项目能找到 clip 包 =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLIP_ROOT = None
for cand in ["clip-main", "CLIP-main"]:
    p = os.path.join(PROJECT_ROOT, cand)
    if os.path.isdir(p):
        CLIP_ROOT = p
        break

if CLIP_ROOT is None:
    raise FileNotFoundError(
        "Cannot find clip-main/ or CLIP-main/ under project root. "
        "Please check your folder name."
    )

if CLIP_ROOT not in sys.path:
    sys.path.insert(0, CLIP_ROOT)

import clip

from src.dataset import PlantWildDataset, build_class_splits
from src.prompt_utils import load_prompt_dict, encode_prompts_per_class_fixed_count
from src.model_utils import GLSimPatchRankLocalAdapter, BoundedAddFusion
from src.utils import set_seed, ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ViT-B/16")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # ===== GLSim + PatchRank local branch =====
    parser.add_argument("--rank_hidden_dim", type=int, default=256)
    parser.add_argument("--rank_dropout", type=float, default=0.1)
    parser.add_argument("--local_hidden_dim", type=int, default=512)
    parser.add_argument("--local_dropout", type=float, default=0.1)
    parser.add_argument("--local_top_r", type=int, default=16)
    parser.add_argument("--local_pool_tau", type=float, default=0.5)
    parser.add_argument("--glsim_weight", type=float, default=0.3)
    parser.add_argument("--rank_weight", type=float, default=0.7)
    parser.add_argument("--max_local_weight", type=float, default=0.10)
    parser.add_argument("--local_init_weight", type=float, default=0.05)

    # ===== single prompt =====
    parser.add_argument("--expected_num_prompts", type=int, default=1)

    # ===== Global-Local/Text calibration =====
    parser.add_argument("--align_eta", type=float, default=0.10,
                        help="penalty weight for global-local disagreement")
    parser.add_argument("--train_with_calibrated_logits", action="store_true",
                        help="if set, train with calibrated logits; otherwise train with fused logits")

    # ===== misc =====
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./outputs/results/glsim_patchrank_gltc")

    return parser.parse_args()


def get_amp_context(device):
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def encode_images_with_patches(model, images, device):
    images = images.to(device, non_blocking=True)

    with get_amp_context(device):
        outputs = model.encode_image(images, return_patch_tokens=True)

    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise RuntimeError(
            "model.encode_image(..., return_patch_tokens=True) must return "
            "(global_feat, patch_tokens). Please check clip/model.py."
        )

    global_features, patch_tokens = outputs

    if patch_tokens is None:
        raise RuntimeError("patch_tokens is None. This script requires a ViT backbone.")

    global_features = F.normalize(global_features.float(), dim=-1)
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    return global_features, patch_tokens


def topk_accuracy(logits, labels, k=1):
    k = min(k, logits.size(1))
    topk_idx = logits.topk(k=k, dim=1).indices
    correct = topk_idx.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct * 100.0


def macro_auc_ovr_safe(y_true, y_score, num_classes):
    auc_list = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(np.int32)
        if y_bin.max() == 0:
            continue
        if y_bin.min() == 1:
            continue
        try:
            auc = roc_auc_score(y_bin, y_score[:, c])
            auc_list.append(auc)
        except Exception:
            continue

    if len(auc_list) == 0:
        return 0.0
    return float(np.mean(auc_list) * 100.0)


def harmonic_mean(seen_acc, unseen_acc, eps=1e-12):
    return float(2.0 * seen_acc * unseen_acc / (seen_acc + unseen_acc + eps))


def evaluate_split(split_name, logits, labels):
    probs = F.softmax(logits, dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    y_pred = logits.argmax(dim=1).cpu().numpy()

    num_classes = logits.size(1)

    result = {
        "split": split_name,
        "top1": round(topk_accuracy(logits, labels, k=1), 2),
        "top3": round(topk_accuracy(logits, labels, k=3), 2),
        "top5": round(topk_accuracy(logits, labels, k=5), 2),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0), 2),
        "macro_auc": round(macro_auc_ovr_safe(y_true, probs, num_classes=num_classes), 2),
    }
    return result


class GLSimPatchRankGLTCModel(torch.nn.Module):
    """
    图像侧：GLSim + PatchRank
    文本侧：single prompt
    第三点：Global-Local / Text Triple Alignment Calibration
    """

    def __init__(
        self,
        feat_dim,
        rank_hidden_dim=256,
        rank_dropout=0.1,
        local_hidden_dim=512,
        local_dropout=0.1,
        local_top_r=16,
        local_pool_tau=0.5,
        glsim_weight=0.3,
        rank_weight=0.7,
        max_local_weight=0.10,
        init_local_weight=0.05,
        align_eta=0.10
    ):
        super().__init__()

        self.local_branch = GLSimPatchRankLocalAdapter(
            feat_dim=feat_dim,
            rank_hidden_dim=rank_hidden_dim,
            rank_dropout=rank_dropout,
            local_hidden_dim=local_hidden_dim,
            local_dropout=local_dropout,
            top_r=local_top_r,
            pool_tau=local_pool_tau,
            glsim_weight=glsim_weight,
            rank_weight=rank_weight
        )

        self.fusion = BoundedAddFusion(
            feat_dim=feat_dim,
            max_local_weight=max_local_weight,
            init_local_weight=init_local_weight
        )

        self.align_eta = align_eta

    def compute_logits(self, global_feats, local_feats, fused_feats, text_bank):
        global_feats = F.normalize(global_feats, dim=-1)
        local_feats = F.normalize(local_feats, dim=-1)
        fused_feats = F.normalize(fused_feats, dim=-1)
        text_bank = F.normalize(text_bank, dim=-1)

        # [B, C]
        global_logits = 100.0 * global_feats @ text_bank.t()
        local_logits = 100.0 * local_feats @ text_bank.t()
        fused_logits = 100.0 * fused_feats @ text_bank.t()

        # disagreement penalty
        disagreement = torch.abs(global_logits - local_logits)  # [B, C]

        # calibrated logits
        calibrated_logits = fused_logits - self.align_eta * disagreement

        return calibrated_logits, fused_logits, global_logits, local_logits, disagreement

    def forward(self, global_feats, patch_tokens, text_bank):
        global_feats = F.normalize(global_feats, dim=-1)

        local_feats, local_aux = self.local_branch(
            global_feats=global_feats,
            patch_tokens=patch_tokens
        )

        fused_feats = self.fusion(global_feats, local_feats)

        calibrated_logits, fused_logits, global_logits, local_logits, disagreement = self.compute_logits(
            global_feats=global_feats,
            local_feats=local_feats,
            fused_feats=fused_feats,
            text_bank=text_bank
        )

        aux = {
            "global_feats": global_feats,
            "local_feats": local_feats,
            "fused_feats": fused_feats,
            "local_weight": self.fusion.get_weight().detach(),
            "fused_logits": fused_logits,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "disagreement": disagreement
        }
        aux.update(local_aux)

        return calibrated_logits, aux


def get_local_weight(model):
    return float(model.fusion.get_weight().detach().cpu().item())


@torch.no_grad()
def collect_logits_for_loader(clip_model, model, loader, text_bank, device):
    clip_model.eval()
    model.eval()

    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        labels = labels.to(device, non_blocking=True)
        global_features, patch_tokens = encode_images_with_patches(clip_model, images, device)
        logits, _ = model(global_features, patch_tokens, text_bank)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels


@torch.no_grad()
def run_full_evaluation(
    clip_model,
    model,
    base_test_loader,
    new_test_loader,
    text_bank,
    device
):
    base_logits, base_labels = collect_logits_for_loader(
        clip_model=clip_model,
        model=model,
        loader=base_test_loader,
        text_bank=text_bank,
        device=device
    )

    new_logits, new_labels = collect_logits_for_loader(
        clip_model=clip_model,
        model=model,
        loader=new_test_loader,
        text_bank=text_bank,
        device=device
    )

    seen_result = evaluate_split("seen(base_test)", base_logits, base_labels)
    unseen_result = evaluate_split("unseen(new_test)", new_logits, new_labels)
    h_score = round(harmonic_mean(seen_result["top1"], unseen_result["top1"]), 2)

    return seen_result, unseen_result, h_score


def format_epoch_result(epoch, epochs, train_loss, seen_result, unseen_result, h_score, local_weight):
    msg = (
        f"[Epoch {epoch:03d}/{epochs:03d}] "
        f"TrainLoss={train_loss:.4f} | "
        f"LocalW={local_weight:.4f} | "
        f"Seen: Top1={seen_result['top1']:.2f} Top3={seen_result['top3']:.2f} "
        f"Top5={seen_result['top5']:.2f} F1={seen_result['macro_f1']:.2f} AUC={seen_result['macro_auc']:.2f} | "
        f"Unseen: Top1={unseen_result['top1']:.2f} Top3={unseen_result['top3']:.2f} "
        f"Top5={unseen_result['top5']:.2f} F1={unseen_result['macro_f1']:.2f} AUC={unseen_result['macro_auc']:.2f} | "
        f"H={h_score:.2f}"
    )
    return msg


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    print("Building class splits...")
    seen_classes, unseen_classes, all_classes = build_class_splits(args.data_root)

    print(f"Seen classes   : {len(seen_classes)}")
    print(f"Unseen classes : {len(unseen_classes)}")
    print(f"All classes    : {len(all_classes)}")

    class_to_global_idx = {name: i for i, name in enumerate(all_classes)}
    seen_global_indices = torch.tensor(
        [class_to_global_idx[name] for name in seen_classes],
        dtype=torch.long,
        device=device
    )

    global_to_seen = torch.full(
        (len(all_classes),),
        fill_value=-1,
        dtype=torch.long,
        device=device
    )
    for local_idx, global_idx in enumerate(seen_global_indices.tolist()):
        global_to_seen[global_idx] = local_idx

    print("Loading CLIP...")
    clip_model, preprocess = clip.load(args.model_name, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if "ViT" not in args.model_name:
        raise ValueError("This script requires a ViT backbone, e.g. ViT-B/16 or ViT-B/32.")

    print("Building datasets...")
    base_train_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "base_train"),
        all_classes=all_classes,
        transform=preprocess
    )
    base_test_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "base_test"),
        all_classes=all_classes,
        transform=preprocess
    )
    new_test_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "new_test"),
        all_classes=all_classes,
        transform=preprocess
    )

    train_loader = DataLoader(
        base_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False
    )
    base_test_loader = DataLoader(
        base_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )
    new_test_loader = DataLoader(
        new_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    print("Loading prompt json...")
    prompt_dict = load_prompt_dict(args.prompt_json)

    print("Encoding single-prompt text bank...")
    all_text_bank = encode_prompts_per_class_fixed_count(
        model=clip_model,
        prompt_dict=prompt_dict,
        classnames=all_classes,
        device=device,
        expected_num_prompts=args.expected_num_prompts
    ).float().to(device)

    if all_text_bank.shape[1] != 1:
        raise ValueError(
            f"This script expects single prompt per class. "
            f"Got expected_num_prompts={args.expected_num_prompts}, actual P={all_text_bank.shape[1]}"
        )

    all_text_bank = all_text_bank[:, 0, :]  # [C, D]
    all_text_bank = F.normalize(all_text_bank, dim=-1)

    seen_text_bank = all_text_bank.index_select(dim=0, index=seen_global_indices)

    feat_dim = all_text_bank.shape[-1]
    print(f"Feature dim: {feat_dim}")
    print(f"Align eta: {args.align_eta}")

    model = GLSimPatchRankGLTCModel(
        feat_dim=feat_dim,
        rank_hidden_dim=args.rank_hidden_dim,
        rank_dropout=args.rank_dropout,
        local_hidden_dim=args.local_hidden_dim,
        local_dropout=args.local_dropout,
        local_top_r=args.local_top_r,
        local_pool_tau=args.local_pool_tau,
        glsim_weight=args.glsim_weight,
        rank_weight=args.rank_weight,
        max_local_weight=args.max_local_weight,
        init_local_weight=args.local_init_weight,
        align_eta=args.align_eta
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    history = []
    best_h = -1.0
    best_epoch = -1
    best_result = None

    print("\n===== Start Training =====")
    for epoch in range(1, args.epochs + 1):
        model.train()
        clip_model.eval()

        running_loss = 0.0
        num_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels_global in pbar:
            labels_global = labels_global.to(device, non_blocking=True)
            labels_local = global_to_seen[labels_global]

            if (labels_local < 0).any():
                bad_idx = torch.where(labels_local < 0)[0][:10].tolist()
                raise RuntimeError(
                    f"Found labels not in seen classes at training time. Example batch indices: {bad_idx}"
                )

            with torch.no_grad():
                global_features, patch_tokens = encode_images_with_patches(clip_model, images, device)

            calibrated_logits, aux = model(
                global_feats=global_features,
                patch_tokens=patch_tokens,
                text_bank=seen_text_bank
            )

            if args.train_with_calibrated_logits:
                train_logits = calibrated_logits
            else:
                train_logits = aux["fused_logits"]

            loss = F.cross_entropy(train_logits, labels_local)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels_global.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            pbar.set_postfix({
                "loss": f"{(running_loss / max(num_samples, 1)):.4f}",
                "lw": f"{get_local_weight(model):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        train_loss = running_loss / max(num_samples, 1)
        local_weight = get_local_weight(model)

        if epoch % args.eval_every == 0:
            seen_result, unseen_result, h_score = run_full_evaluation(
                clip_model=clip_model,
                model=model,
                base_test_loader=base_test_loader,
                new_test_loader=new_test_loader,
                text_bank=all_text_bank,
                device=device
            )

            epoch_record = {
                "epoch": epoch,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
                "train_loss": round(float(train_loss), 6),
                "local_weight": round(float(local_weight), 6),
                "seen": seen_result,
                "unseen": unseen_result,
                "h_score": h_score
            }
            history.append(epoch_record)

            print(format_epoch_result(
                epoch=epoch,
                epochs=args.epochs,
                train_loss=train_loss,
                seen_result=seen_result,
                unseen_result=unseen_result,
                h_score=h_score,
                local_weight=local_weight
            ))

            history_path = os.path.join(args.output_dir, "history.json")
            save_json(history, history_path)

            if h_score > best_h:
                best_h = h_score
                best_epoch = epoch
                best_result = epoch_record

                ckpt_path = os.path.join(args.output_dir, "best_h_glsim_patchrank_gltc.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_h": best_h,
                    "args": vars(args)
                }, ckpt_path)

                best_json_path = os.path.join(args.output_dir, "best_h_metrics.json")
                save_json(best_result, best_json_path)

    print("\n===== Training Finished =====")
    if best_result is None:
        print("No evaluation result found. Please check eval_every / epochs.")
        return

    final_result = {
        "method": "glsim_patchrank_gltc",
        "model_name": args.model_name,
        "prompt_json": args.prompt_json,
        "expected_num_prompts": args.expected_num_prompts,
        "align_eta": args.align_eta,
        "train_with_calibrated_logits": args.train_with_calibrated_logits,
        "rank_hidden_dim": args.rank_hidden_dim,
        "rank_dropout": args.rank_dropout,
        "local_hidden_dim": args.local_hidden_dim,
        "local_dropout": args.local_dropout,
        "local_top_r": args.local_top_r,
        "local_pool_tau": args.local_pool_tau,
        "glsim_weight": args.glsim_weight,
        "rank_weight": args.rank_weight,
        "max_local_weight": args.max_local_weight,
        "local_init_weight": args.local_init_weight,
        "num_seen_classes": len(seen_classes),
        "num_unseen_classes": len(unseen_classes),
        "num_all_classes": len(all_classes),
        "best_epoch": best_epoch,
        "best_result": best_result
    }

    final_json = os.path.join(args.output_dir, "final_result.json")
    save_json(final_result, final_json)

    print("\n===== Best H-Score Result =====")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    print(f"Saved to: {final_json}")


if __name__ == "__main__":
    main()