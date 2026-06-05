import os


CHECKPOINT_DIR = os.getenv("CYG_CHECKPOINT_DIR", "_cyg_experiment_checkpoints")

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"
BACKBONE_VSSM_SMALL = "vssm_small"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
    BACKBONE_VSSM_SMALL,
]

BATCH_SIZE = int(os.getenv("CYG_BATCH_SIZE", "16"))
NUM_EPOCHS = int(os.getenv("CYG_NUM_EPOCHS", "500"))
LR = float(os.getenv("CYG_LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("CYG_WEIGHT_DECAY", "1e-3"))
GRAD_CLIP_NORM = float(os.getenv("CYG_GRAD_CLIP_NORM", "0.0"))
EARLY_STOP_PATIENCE = int(os.getenv("CYG_EARLY_STOP_PATIENCE", "0"))
EARLY_STOP_MIN_DELTA = float(os.getenv("CYG_EARLY_STOP_MIN_DELTA", "0.0"))

LOG_INTERVAL = int(os.getenv("CYG_LOG_INTERVAL", "10"))
EVAL_INTERVAL = int(os.getenv("CYG_EVAL_INTERVAL", "10"))
CHECKPOINT_INTERVAL = int(os.getenv("CYG_CHECKPOINT_INTERVAL", "10"))
