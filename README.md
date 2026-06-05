# CyG-Flow


![](figures/pipeline.png)

**CyG-Flow: Cyclic Receptive Field Transformation with Gated Multi-scale Normalizing Flows for Industrial Anomaly Detection**

[Paper link]()

## Introduction

PyTorch implementation of **CyG-Flow** for unsupervised anomaly detection on **MVTec AD**, with a frozen **Vision Mamba** encoder (`vssm_small`).

- **Cyclic receptive field transformation** in flow coupling layers.
- **Gated multi-scale fusion** with reversible channel shuffle.
- **Adaptive Statistical Energy Scoring** for image-level detection (config: `gfn_mix`).

Pipeline overview: frozen backbone → normalizing flow (CRF-B subnets) → gated fusion → scoring (see figure above). CRF-B follows \(1\times1 \rightarrow 3\times3 \rightarrow 5\times5\) cyclic receptive fields; in code the \(5\times5\) step is implemented as two stacked \(3\times3\) convolutions (equivalent effective receptive field, see `nf_cyg_flow_vmamba` in `vmamba_flow.py`).

## Get Started

> **Not included in this repo:** only CyG-Flow source code and configs are shipped here. After `git clone`, you must **separately** clone [VMamba](https://github.com/MzeroMiko/VMamba) into `VMamba/` and **download** the `vssm_small` checkpoint (steps 3–4 below). Training and evaluation will fail without both.

### Environment

**Python 3.10+**. Optional Conda env:

```bash
conda create -n cygflow python=3.10 -y
conda activate cygflow
```

**1. PyTorch** — install a build that matches your CUDA (example: CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Other CUDA versions: [pytorch.org](https://pytorch.org).

**2. Python dependencies** — from repo root `CyG-Flow/`:

```bash
pip install -r requirements.txt
```

Pinned packages are listed in `requirements.txt` (`timm==0.5.4`, `FrEIA`, `pyyaml`, `scikit-learn`, `fvcore`, etc.).

**FrEIA** — if `pip install -r requirements.txt` fails on git / GitHub timeout:

```bash
pip install "https://github.com/VLL-HD/FrEIA/archive/1779d1fba1e21000fda1927b59eeac0a6fcaa284.tar.gz"
pip install timm==0.5.4 "pyyaml>=6.0" "scikit-learn>=1.0.0" "numpy>=1.21.0" "Pillow>=9.0.0" fvcore packaging
```

**3. VMamba** — clone under `CyG-Flow/VMamba/` and build `selective_scan` on Linux when using the CUDA extension (see [VMamba](https://github.com/MzeroMiko/VMamba)):

```bash
cd CyG-Flow
git clone https://github.com/MzeroMiko/VMamba.git VMamba
```

On Windows you can use a directory junction: `mklink /D VMamba D:\path\to\VMamba`.

**4. VMamba pretrained weights (required)** — CyG-Flow uses a **frozen VMamba** encoder (`vssm_small`, ImageNet-1K). The checkpoint is **not** included in this repo; download the official VMamba release and save as:

```
CyG-Flow/vim_small_midclstok/vssm_small_0229_ckpt_epoch_222.pth
```

| Item | URL |
|------|-----|
| **vssm_small checkpoint (default)** | https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth |
| All VMamba classification weights | https://github.com/MzeroMiko/VMamba#classification-on-imagenet-1k |

```bash
cd CyG-Flow
mkdir -p vim_small_midclstok
wget -O vim_small_midclstok/vssm_small_0229_ckpt_epoch_222.pth \
  "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth"
```

Or set an absolute path in the config:

```yaml
vssm_ckpt: /path/to/CyG-Flow/vim_small_midclstok/vssm_small_0229_ckpt_epoch_222.pth
```

### Data

#### MVTec AD

Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

Use the original folder layout (`<category>/train/good/`, `<category>/test/`, etc.). Pass the dataset root to `--data`.

### Reproducibility

Training uses `--seed` (default `42`) and an optional `--deterministic` flag. For stricter, repeatable runs (slower), pass both explicitly:

```bash
--seed 42 --deterministic
```

Hyperparameters are fixed in `configs/vssm_small_vmamba.yaml`. Match your PyTorch/CUDA build to the versions you used when reporting numbers.

### Run

#### Train (single category)

```bash
cd CyG-Flow
python main_vmamba.py \
  -cfg configs/vssm_small_vmamba.yaml \
  --data /path/to/mvtec \
  -cat bottle \
  --gpu 0 \
  --seed 42 \
  --deterministic \
  --results-csv vmamba_mvtec_summary.csv
```

#### Train (all 15 categories)

```bash
python main_vmamba.py \
  -cfg configs/vssm_small_vmamba.yaml \
  --data /path/to/mvtec \
  -cat all \
  --gpu 0 \
  --seed 42 \
  --deterministic \
  --results-csv vmamba_mvtec_summary.csv
```

#### Eval only

```bash
python main_vmamba.py \
  -cfg configs/vssm_small_vmamba.yaml \
  --data /path/to/mvtec \
  -cat bottle \
  --gpu 0 \
  --eval
```

Checkpoints and CSV summaries are written under `CYG_CHECKPOINT_DIR` (default: `_cyg_experiment_checkpoints`) and `_bg_runs/vmamba_per_category/`.


## Acknowledgement

Thanks for inspiration from [VMamba](https://github.com/MzeroMiko/VMamba), [FrEIA](https://github.com/VLL-HD/FrEIA), and receptive-field block designs used in CRFBlock.

## License

All code in this repository is under the [MIT license](LICENSE).
