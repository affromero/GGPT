# GGPT: Geometry-grounded Point Transformer

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](#)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/YutongGoose/GGPT)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-green)](https://huggingface.co/datasets/YutongGoose/GGPT_eval)

Official PyTorch implementation of **Geometry-grounded Point Transformer (GGPT)** (CVPR 2026). GGPT is a method for high-quality 3D reconstruction and point-cloud understanding from multiview images.

---

## 🚀 Pretrained Models

You can download our pretrained checkpoint directly from Hugging Face: [YutongGoose/GGPT](https://huggingface.co/YutongGoose/GGPT).

```bash
# Recommended: download via huggingface-cli
pip install -U "huggingface_hub[cli]"
huggingface-cli download YutongGoose/GGPT model.step228000.pth --local-dir ckpts/
```

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone --recursive https://github.com/your-username/GGPT.git
cd GGPT
```


### 2. Install dependencies

#### 2.0 Create a virtual environment.
#### 2.1 Install torch, torchvision for your CUDA version. (The environment does not require specific CUDA or pytorch version. It has been tested in CUDA 12.1/12.3 and torch 2.2.0/2.5.1.)

#### 2.2 Install requirements for VGGT and SfM.
```
pip install -r requirements_sfm.txt
cd RoMAv2/ && pip install -e .
```

#### 2.3 \[Optional\] If you need to run GGPT, the 3D point transformer, please install the following packages in the same virtual environement.  You don't need to build another env for this.

```
pip install torch-sparse  # (add  `--no-build-isolation' if needed)
pip install torch-scatter # --no-build-isolation
pip install torch-geometric # --no-build-isolation
pip install torch-cluster # --no-build-isolation
cd Pointcept/ pip install -e .
pip install spconv-cu121 # Ensure it matches your PyTorch CUDA version.
pip install flash-attn --no-build-isolation 
# Build flash-attn from source if needed. For example:
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --no-build-isolation --no-cache-dir flash-attn
```
Note that the recipe of Ptv3 installation's highly depends on your pytorch and cuda version. You need to change the version of spconv and flash-attention accordingly, or install them from source when needed.

## ⚙️ Configuration Settings (SfM)

You can adjust Structure-from-Motion (SfM) configuration blocks in the `.yaml` files (found under `configs/`) to better suit your data. 

```yaml
ba_config:
  shared_camera: True # Set it to False if images are captured with different camera intrinsics

dlt_config:
  # Adjust the filtering parameters if you need more accurate yet sparser SfM points. E.g.:
  score_thresh: 0.1       # Increase the matching confidence threshold to filter out noisy matchings
  cycle_err_thresh: 4     # Reduce the cycle error threshold to filter out noisy tracks
  max_epipolar_error: 4   # Reduce the epipolar error threshold to filter out noisy tracks
  min_tri_angle: 3        # Increase the triangulation angle threshold to filter out points with low parallax
  max_reproj_error: 4     # Reduce the reprojection error threshold to filter out noisy points
```

---

## 📊 Evaluation

To run evaluations on the benchmark:

**1. Download the Dataset:**
Download the [preprocessed evaluation set](https://huggingface.co/datasets/YutongGoose/GGPT_eval) and place it at the root of this project as `GGPT_eval/`.

**2. Run Evaluation:**
```bash
sh benchmark_eval.sh
```
