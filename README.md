# PlantWild Single-Prompt GZSL

This project focuses on **Generalized Zero-Shot Learning (GZSL) for plant disease recognition** based on CLIP under a single-prompt setting. To address the insufficient representation of local lesion regions and the seen-class bias in plant disease images, the framework is built around the following three core modules:

- **Module 1: Global-guided Local Enhancement (GLE) Module**
- **Module 2: Global-Local/Text Alignment Calibration (GLTC)**
- **Module 3: Seen Temperature Scaling (STS)**

In particular:

- **GLE** enhances lesion-related local visual representations and fuses them with global image features;
- **GLTC** calibrates classification logits based on the alignment relationships among global features, local features, and text features;
- **STS** is applied only during evaluation to scale the logits of seen classes, thereby alleviating the common seen-class bias in GZSL.

---

## 1. Repository Structure

```text
plantwild_single_gzsl_code/
├─ CLIP-main/
│  └─ ...                                 # Official CLIP source code; only model.py is modified
├─ configs/
│  ├─ glsim_patchrank_gltc_train.yaml
│  └─ glsim_patchrank_gltc_sts_eval.yaml
├─ prompts/
│  └─ plantwild_single_prompt.json
├─ src/
│  ├─ dataset.py
│  ├─ model_utils.py
│  ├─ prompt_utils.py
│  └─ utils.py
├─ train_glsim_patchrank_gltc.py
├─ eval_glsim_patchrank_gltc_sts.py
└─ requirements.txt
````

The components are summarized as follows:

* `CLIP-main/`: The CLIP source directory downloaded from the official GitHub repository. The original folder structure is preserved, and only `model.py` is modified as required by this project.
* `configs/`: Stores the training and evaluation configurations used in the main experiments.
* `prompts/`: Stores prompt files. This project currently uses a single-prompt setting.
* `src/dataset.py`: Dataset loading, class split construction, and GZSL data organization.
* `src/model_utils.py`: Core implementation of the GLE-related modules.
* `src/prompt_utils.py`: Prompt processing and text-feature-related utilities.
* `src/utils.py`: General utility functions.
* `train_glsim_patchrank_gltc.py`: Training script for GLE + GLTC.
* `eval_glsim_patchrank_gltc_sts.py`: Evaluation script that loads a trained checkpoint and applies STS during inference.

---

## 2. Environment and Installation

The reference environment for this project is:

* Python 3.10
* PyTorch 2.5.1
* torchvision 0.20.1
* CUDA 12.1

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 3. Modified CLIP Backbone

This project uses the `CLIP-main` directory downloaded from the official GitHub repository and keeps its original folder name.

To support local lesion modeling, we modify `model.py` so that the ViT image encoder supports:

* `return_patch_tokens=True`

This allows the model to return both:

* global image features
* patch-level token representations

These patch-level tokens are required by the **Global-guided Local Enhancement (GLE) Module**.

Therefore, the modified `model.py` in this project should **not** be replaced with the unmodified original CLIP version; otherwise, the related code will not run properly.

---

## 4. Data Preparation

The expected dataset directory structure is:

```text
datasets/plantwild/
├─ base_train/
├─ base_test/
└─ new_test/
```

where:

* `base_train/`: seen-class training set
* `base_test/`: seen-class test set
* `new_test/`: unseen-class test set

Please organize the dataset in the above format before running the training or evaluation scripts.

---

## 5. Prompt File

This project currently uses a **single-prompt** setting with the following prompt file:

```text
prompts/plantwild_single_prompt.json
```

In this setting:

* each class uses one prompt
* therefore, `expected_num_prompts=1`

If multi-prompt experiments are added later, corresponding JSON files can be placed in `prompts/` and the related arguments can be adjusted accordingly.

---

## 6. Config Files

The `configs/` directory contains two configuration files for the main experiments:

* `glsim_patchrank_gltc_train.yaml`: training configuration
* `glsim_patchrank_gltc_sts_eval.yaml`: evaluation configuration

These files record the final parameter settings used in the main experiments for reproducibility and management. Even if the scripts are run mainly through command-line arguments, the YAML files still serve as the reference configurations.

---

## 7. Method Overview

### Module 1: Global-guided Local Enhancement (GLE) Module

The **Global-guided Local Enhancement (GLE) Module** is used to enhance local lesion-related representations on the image side and fuse them with global semantic features. Its main idea is:

1. extract global features and patch-level tokens from the modified CLIP image encoder;
2. use a global-guided local scoring mechanism to identify potentially informative lesion regions;
3. aggregate the selected local patch features;
4. fuse the enhanced local representation with the global representation.

---

### Module 2: Global-Local/Text Alignment Calibration (GLTC)

**Global-Local/Text Alignment Calibration (GLTC)** calibrates classification logits based on the alignment relationships among global features, local features, and class text features.

By leveraging the response discrepancy between the global branch and the local branch in the text feature space, GLTC adjusts the fused logits to reduce representation bias and improve classification discrimination.

---

### Module 3: Seen Temperature Scaling (STS)

**Seen Temperature Scaling (STS)** is used only during evaluation.

Its purpose is to scale the logits of seen classes so as to alleviate the seen-class bias commonly observed in generalized zero-shot learning, thereby improving the balance between seen and unseen performance.

---

## 8. Training

The training script is:

```text
train_glsim_patchrank_gltc.py
```

This script is used to train:

* **GLE**
* **GLTC**

An example training command is:

```bash
python train_glsim_patchrank_gltc.py \
  --data_root ./datasets/plantwild \
  --prompt_json ./prompts/plantwild_single_prompt.json \
  --model_name ViT-B/16 \
  --batch_size 8 \
  --num_workers 4 \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --rank_hidden_dim 256 \
  --rank_dropout 0.1 \
  --local_hidden_dim 512 \
  --local_dropout 0.1 \
  --local_top_r 16 \
  --local_pool_tau 0.5 \
  --glsim_weight 0.3 \
  --rank_weight 0.7 \
  --max_local_weight 0.10 \
  --local_init_weight 0.05 \
  --expected_num_prompts 1 \
  --align_eta 0.05 \
  --output_dir ./outputs/results/glsim_gltc_eta005
```

The main outputs during training typically include:

* the best checkpoint
* training logs
* validation/evaluation results

Please refer to the actual saving logic in the code for the exact output files.

---

## 9. Evaluation

The evaluation script is:

```text
eval_glsim_patchrank_gltc_sts.py
```

This script is used to:

1. load a trained checkpoint;
2. perform inference with GLE + GLTC;
3. apply **STS** during evaluation to obtain the final GZSL results.

An example evaluation command is:

```bash
python eval_glsim_patchrank_gltc_sts.py \
  --data_root ./datasets/plantwild \
  --prompt_json ./prompts/plantwild_single_prompt.json \
  --model_name ViT-B/16 \
  --batch_size 8 \
  --num_workers 4 \
  --rank_hidden_dim 256 \
  --rank_dropout 0.1 \
  --local_hidden_dim 512 \
  --local_dropout 0.1 \
  --local_top_r 16 \
  --local_pool_tau 0.5 \
  --glsim_weight 0.3 \
  --rank_weight 0.7 \
  --max_local_weight 0.10 \
  --local_init_weight 0.05 \
  --expected_num_prompts 1 \
  --align_eta 0.05 \
  --seen_temperature 1.05 \
  --resume_ckpt ./outputs/results/glsim_gltc_eta005/best_h_glsim_patchrank_gltc.pt \
  --output_dir ./outputs/results/gltc_sts_t105
```

Notes:

* `seen_temperature` is used only during evaluation;
* the evaluation script requires a trained checkpoint in advance;
* `resume_gltc_ckpt` should point to the checkpoint saved during training.

---

## 10. Key Arguments

Some important arguments are listed below:

* `local_top_r`: number of selected local patches
* `local_pool_tau`: temperature parameter for soft pooling during local feature aggregation
* `glsim_weight`: weight of the global-guided local scoring branch in GLE
* `rank_weight`: weight of the patch-ranking branch in GLE
* `max_local_weight`: maximum fusion weight for local features
* `local_init_weight`: initial fusion weight for local features
* `align_eta`: calibration coefficient in GLTC for global/local-text alignment discrepancy
* `seen_temperature`: temperature parameter used in STS for seen-class logit scaling

---

## 11. Output Files

Training and evaluation results are saved according to `--output_dir`.

Typical outputs include:

* checkpoints saved during training
* evaluation result files
* logs or intermediate records

It is recommended to use different `output_dir` values for different experiments for easier result management.

---

## 12. Acknowledgement and License

This project is developed based on OpenAI CLIP and keeps the original source directory as `CLIP-main/`.

Only `model.py` is modified as required by the proposed method to support patch-level token output.

If you use this project, please also follow the license requirements of the original CLIP project and acknowledge the original source where appropriate.

---
