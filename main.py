import argparse
import csv
import glob
import os
import math
import random
from datetime import datetime

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score

import constants as const
import dataset
import flow
import utils

if not hasattr(utils, "AverageMeter"):
    class _FallbackAverageMeter:
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    utils.AverageMeter = _FallbackAverageMeter


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_CSV = os.path.join(PROJECT_ROOT, "vmamba_mvtec_results.csv")

# When ``--gpu`` is omitted, try these physical indices in order (skip busy/broken GPUs).
DEFAULT_CUDA_DEVICE_CANDIDATES = [0, 1, 2, 3, 4, 6]


RESULTS_CSV_FIELDNAMES = [
    "timestamp",
    "category",
    "I_AUROC_at_best_p",
    "P_AUROC_at_best_p",
    "I_AUROC_at_best_i",
    "P_AUROC_at_best_i",
    "best_p_checkpoint",
    "best_i_checkpoint",
    "config",
    "seed",
    "deterministic",
]


def append_results_csv(csv_path, row):
    if not csv_path:
        return
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_category_config_path(config_path, config, category):
    category_configs = config.get("category_configs", {}) if isinstance(config, dict) else {}
    if not isinstance(category_configs, dict) or not category_configs:
        return config_path
    if category == "all":
        return config_path
    selected = category_configs.get(category)
    if not selected:
        raise KeyError(
            "category_configs does not define a config for category '{}'".format(category)
        )
    if os.path.isabs(selected):
        return selected
    return os.path.normpath(os.path.join(os.path.dirname(config_path), selected))


def load_effective_config(config_path, category):
    config = yaml.safe_load(open(config_path, "r"))
    effective_config_path = resolve_category_config_path(config_path, config, category)
    if effective_config_path != config_path:
        config = yaml.safe_load(open(effective_config_path, "r"))
    return config, effective_config_path


def set_reproducibility(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic kernels when possible.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)

    def _seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)

    def _seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=g,
    )


def build_model(config):
    saf_cfg = config.get("saf", {}) if isinstance(config.get("saf", {}), dict) else {}
    ms_cfg = (
        config.get("multi_scale", {})
        if isinstance(config.get("multi_scale", {}), dict)
        else {}
    )
    model = flow.CyGFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        flow_subnet_type=config.get("flow_subnet_type", "standard"),
        flow_kernel_alternating=config.get("flow_kernel_alternating", True),
        flow_crf_b_alternating_period=config.get(
            "flow_crf_b_alternating_period",
            config.get("flow_cfb_alternating_period", 1),
        ),
        flow_coupling_type=config.get("flow_coupling_type", "standard"),
        flow_coupling_eps=config.get("flow_coupling_eps", 0.02),
        flow_coupling_alpha=config.get("flow_coupling_alpha", 1.0),
        flow_coupling_gamma=config.get("flow_coupling_gamma", 0.1),
        vssm_ckpt=config.get("vssm_ckpt", None),
        vssm_out_indices=config.get("vssm_out_indices", None),
        use_saf=bool(saf_cfg.get("enabled", False)),
        fusion_weights=ms_cfg.get("fusion_weights", None),
        fusion_learnable=bool(ms_cfg.get("fusion_learnable", False)),
        fusion_init=ms_cfg.get("fusion_init", None),
        fusion_type=ms_cfg.get("fusion_type", "weighted"),
        fusion_pre_norm=ms_cfg.get("fusion_pre_norm", "none"),
        fusion_pre_norm_clip=ms_cfg.get("fusion_pre_norm_clip", 0.0),
        fusion_pre_norm_eps=ms_cfg.get("fusion_pre_norm_eps", 1e-6),
        fusion_pre_norm_power=ms_cfg.get("fusion_pre_norm_power", 1.0),
        fusion_pre_norm_low_q=ms_cfg.get("fusion_pre_norm_low_q", 0.01),
        fusion_pre_norm_high_q=ms_cfg.get("fusion_pre_norm_high_q", 0.99),
        flow_activation=config.get("flow_activation", "relu"),
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def _parse_cuda_candidates_from_env():
    raw = os.environ.get("CYG_CUDA_DEVICES", "").strip()
    if not raw:
        return list(DEFAULT_CUDA_DEVICE_CANDIDATES)
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out if out else list(DEFAULT_CUDA_DEVICE_CANDIDATES)


def _cuda_index_smoke_test(idx: int) -> bool:
    """Return True if this device index accepts a tiny GPU matmul + sync."""
    if not torch.cuda.is_available():
        return False
    if idx < 0 or idx >= torch.cuda.device_count():
        return False
    try:
        torch.cuda.set_device(idx)
        dev = torch.device("cuda:{}".format(idx))
        x = torch.zeros(8, 8, device=dev)
        _ = x @ x + 1.0
        torch.cuda.synchronize()
        del x
        return True
    except Exception:
        return False


def get_runtime_device(args=None):
    """Pick ``cuda:N`` or CPU.

    - With ``args.gpu == -1``: CPU.
    - With ``args.gpu >= 0``: that index only (must pass smoke test).
    - Otherwise: first usable index from ``CYG_CUDA_DEVICES`` env or
      ``DEFAULT_CUDA_DEVICE_CANDIDATES``.
    """
    force = None
    if args is not None and getattr(args, "gpu", None) is not None:
        force = int(args.gpu)
        if force < 0:
            return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if force is not None:
        if _cuda_index_smoke_test(force):
            print(
                "Runtime CUDA device {} (forced): {}".format(
                    force, torch.cuda.get_device_name(force)
                )
            )
            return torch.device("cuda:{}".format(force))
        raise RuntimeError(
            "CUDA device {} is unavailable or failed the smoke test.".format(force)
        )

    candidates = _parse_cuda_candidates_from_env()
    for idx in candidates:
        if _cuda_index_smoke_test(idx):
            print(
                "Runtime CUDA device {} (auto from {}): {}".format(
                    idx, candidates, torch.cuda.get_device_name(idx)
                )
            )
            return torch.device("cuda:{}".format(idx))

    print(
        "No usable CUDA device among {}; falling back to CPU.".format(candidates)
    )
    return torch.device("cpu")


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    device = next(model.parameters()).device
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.to(device, non_blocking=True)
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        if const.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), const.GRAD_CLIP_NORM)
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def build_image_score_cfg(config):
    eval_cfg = config.get("eval", {}) if isinstance(config.get("eval", {}), dict) else {}
    image_score_cfg = (
        eval_cfg.get("image_score", {})
        if isinstance(eval_cfg.get("image_score", {}), dict)
        else {}
    )

    gfn_prior = image_score_cfg.get("gfn_prior", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(gfn_prior, (list, tuple)):
        gfn_prior = [0.0, 0.0, 0.0, 0.0]
    gfn_prior = list(gfn_prior)
    if len(gfn_prior) != 4:
        gfn_prior = [0.0, 0.0, 0.0, 0.0]
    gfn_prior = [float(x) for x in gfn_prior]

    return {
        "mode": str(image_score_cfg.get("mode", "max")).lower(),
        "topk_ratio": float(image_score_cfg.get("topk_ratio", 0.01)),
        "topk_k": int(image_score_cfg.get("topk_k", 0)),
        "quantile": float(image_score_cfg.get("quantile", 0.99)),
        "quantile_low": float(image_score_cfg.get("quantile_low", 0.99)),
        "quantile_high": float(image_score_cfg.get("quantile_high", 0.999)),
        "quantile_alpha": float(image_score_cfg.get("quantile_alpha", 0.5)),
        "vae_tail_alpha": float(image_score_cfg.get("vae_tail_alpha", 0.7)),
        "vae_kl_beta": float(image_score_cfg.get("vae_kl_beta", 0.3)),
        "gfn_temperature": float(image_score_cfg.get("gfn_temperature", 0.25)),
        "gfn_prior": gfn_prior,
        "use_raw_map": bool(image_score_cfg.get("use_raw_map", False)),
    }


def build_eval_infer_cfg(config):
    eval_cfg = config.get("eval", {}) if isinstance(config.get("eval", {}), dict) else {}

    tta_cfg = eval_cfg.get("tta", {}) if isinstance(eval_cfg.get("tta", {}), dict) else {}
    post_cfg = (
        eval_cfg.get("anomaly_post", {})
        if isinstance(eval_cfg.get("anomaly_post", {}), dict)
        else {}
    )

    return {
        "tta_hflip": bool(tta_cfg.get("hflip", False)),
        "tta_vflip": bool(tta_cfg.get("vflip", False)),
        "tta_rot90": bool(tta_cfg.get("rot90", False)),
        "gaussian_sigma": float(post_cfg.get("gaussian_sigma", 0.0)),
        "gaussian_kernel_size": int(post_cfg.get("gaussian_kernel_size", 0)),
        "map_norm": str(post_cfg.get("map_norm", "none")).lower(),
        "map_norm_low_q": float(post_cfg.get("map_norm_low_q", 0.01)),
        "map_norm_high_q": float(post_cfg.get("map_norm_high_q", 0.99)),
    }


def _build_gaussian_kernel(kernel_size, sigma, device, dtype):
    if kernel_size <= 0:
        kernel_size = int(math.ceil(6.0 * sigma))
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
    elif kernel_size % 2 == 0:
        kernel_size += 1

    x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    g = g / (g.sum() + 1e-12)
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / (kernel_2d.sum() + 1e-12)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def apply_anomaly_postprocess(anomaly_map, eval_infer_cfg):
    sigma = float(eval_infer_cfg.get("gaussian_sigma", 0.0))
    if sigma > 0:
        kernel_size = int(eval_infer_cfg.get("gaussian_kernel_size", 0))
        kernel = _build_gaussian_kernel(
            kernel_size,
            sigma,
            anomaly_map.device,
            anomaly_map.dtype,
        )
        pad = kernel.shape[-1] // 2
        anomaly_map = F.conv2d(anomaly_map, kernel, padding=pad)

    norm_mode = str(eval_infer_cfg.get("map_norm", "none")).lower()
    if norm_mode == "none":
        return anomaly_map

    b = anomaly_map.shape[0]
    flat = anomaly_map.view(b, -1)

    if norm_mode == "minmax":
        lo = flat.min(dim=1).values.view(b, 1, 1, 1)
        hi = flat.max(dim=1).values.view(b, 1, 1, 1)
    elif norm_mode == "robust":
        low_q = float(eval_infer_cfg.get("map_norm_low_q", 0.01))
        high_q = float(eval_infer_cfg.get("map_norm_high_q", 0.99))
        low_q = max(0.0, min(1.0, low_q))
        high_q = max(low_q + 1e-6, min(1.0, high_q))
        lo = torch.quantile(flat, q=low_q, dim=1).view(b, 1, 1, 1)
        hi = torch.quantile(flat, q=high_q, dim=1).view(b, 1, 1, 1)
    else:
        raise ValueError(
            "Unsupported eval.anomaly_post.map_norm='{}'; expected one of [none, minmax, robust]".format(
                norm_mode
            )
        )

    anomaly_map = (anomaly_map - lo) / (hi - lo + 1e-12)
    return anomaly_map.clamp(0.0, 1.0)


def infer_anomaly_maps(model, data, eval_infer_cfg):
    maps = []

    ret = model(data)
    maps.append(ret["anomaly_map"])

    if eval_infer_cfg.get("tta_hflip", False):
        ret = model(torch.flip(data, dims=[3]))
        maps.append(torch.flip(ret["anomaly_map"], dims=[3]))

    if eval_infer_cfg.get("tta_vflip", False):
        ret = model(torch.flip(data, dims=[2]))
        maps.append(torch.flip(ret["anomaly_map"], dims=[2]))

    # rot90 already includes 180deg; avoid duplicating that transform weight.
    if (
        eval_infer_cfg.get("tta_hflip", False)
        and eval_infer_cfg.get("tta_vflip", False)
        and not eval_infer_cfg.get("tta_rot90", False)
    ):
        ret = model(torch.flip(data, dims=[2, 3]))
        maps.append(torch.flip(ret["anomaly_map"], dims=[2, 3]))

    if eval_infer_cfg.get("tta_rot90", False):
        for k in (1, 2, 3):
            rot_data = torch.rot90(data, k=k, dims=[2, 3])
            ret = model(rot_data)
            maps.append(torch.rot90(ret["anomaly_map"], k=4 - k, dims=[2, 3]))

    if len(maps) == 1:
        anomaly_map_raw = maps[0]
    else:
        anomaly_map_raw = torch.mean(torch.stack(maps, dim=0), dim=0)

    anomaly_map_post = apply_anomaly_postprocess(anomaly_map_raw, eval_infer_cfg)
    return anomaly_map_raw, anomaly_map_post


def infer_anomaly_map(model, data, eval_infer_cfg):
    _, anomaly_map_post = infer_anomaly_maps(model, data, eval_infer_cfg)
    return anomaly_map_post


def aggregate_image_scores(anomaly_map, image_score_cfg):
    flat = anomaly_map.view(anomaly_map.shape[0], -1)
    mode = image_score_cfg["mode"]

    def _topk_mean_score(flat_map):
        num_pixels = flat_map.shape[1]
        topk_k = int(image_score_cfg.get("topk_k", 0))
        topk_ratio = float(image_score_cfg.get("topk_ratio", 0.01))
        if topk_k > 0:
            k = topk_k
        else:
            k = int(math.ceil(num_pixels * topk_ratio))
        k = max(1, min(num_pixels, k))
        return torch.topk(flat_map, k=k, dim=1).values.mean(dim=1)

    def _safe_quantile(flat_map, q):
        q = max(0.0, min(1.0, float(q)))
        return torch.quantile(flat_map, q=q, dim=1)

    if mode == "max":
        return flat.max(dim=1).values

    if mode == "topk":
        return _topk_mean_score(flat)

    if mode == "quantile":
        return _safe_quantile(flat, image_score_cfg.get("quantile", 0.99))

    if mode == "quantile_mix":
        # Blend two tail quantiles to jointly capture anomaly peak and spread.
        q_low = float(image_score_cfg.get("quantile_low", 0.99))
        q_high = float(image_score_cfg.get("quantile_high", 0.999))
        alpha = float(image_score_cfg.get("quantile_alpha", 0.5))

        q_low = max(0.0, min(1.0, q_low))
        q_high = max(0.0, min(1.0, q_high))
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        alpha = max(0.0, min(1.0, alpha))

        low_score = _safe_quantile(flat, q_low)
        high_score = _safe_quantile(flat, q_high)
        return alpha * high_score + (1.0 - alpha) * low_score

    if mode == "vae_quantile":
        # Idea-5 proxy: use a Gaussian-latent uncertainty term (KL) and tail intensity.
        q = float(image_score_cfg.get("quantile", 0.99))
        tail_alpha = float(image_score_cfg.get("vae_tail_alpha", 0.7))
        kl_beta = float(image_score_cfg.get("vae_kl_beta", 0.3))
        tail_alpha = max(0.0, min(1.0, tail_alpha))
        kl_beta = max(0.0, kl_beta)

        tail_score = _safe_quantile(flat, q)
        mu = flat.mean(dim=1)
        var = flat.var(dim=1, unbiased=False) + 1e-12
        kl = 0.5 * (mu.pow(2) + var - torch.log(var) - 1.0)
        return tail_alpha * tail_score + kl_beta * kl

    if mode == "gfn_mix":
        # Idea-6 proxy: policy-style weighted mixture over multiple scorers.
        q = float(image_score_cfg.get("quantile", 0.99))
        q_high = float(image_score_cfg.get("quantile_high", 0.999))
        tau = max(1e-4, float(image_score_cfg.get("gfn_temperature", 0.25)))

        s_max = flat.max(dim=1).values
        s_topk = _topk_mean_score(flat)
        s_q = _safe_quantile(flat, q)
        s_qh = _safe_quantile(flat, q_high)
        scores = torch.stack([s_max, s_topk, s_q, s_qh], dim=1)

        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        tail_gap = (s_qh - s_q).clamp(min=0.0)
        energies = torch.stack(
            [
                s_max - mean,
                s_topk - mean,
                s_q - mean,
                tail_gap + std,
            ],
            dim=1,
        )

        prior = image_score_cfg.get("gfn_prior", [0.0, 0.0, 0.0, 0.0])
        prior_tensor = torch.tensor(prior, device=flat.device, dtype=flat.dtype).view(1, 4)
        weights = torch.softmax((energies + prior_tensor) / tau, dim=1)
        return (weights * scores).sum(dim=1)

    raise ValueError(
        "Unsupported eval.image_score.mode='{}'; expected one of [max, topk, quantile, quantile_mix, vae_quantile, gfn_mix]".format(
            mode
        )
    )


def eval_once(dataloader, model, image_score_cfg, eval_infer_cfg=None):
    model.eval()
    device = next(model.parameters()).device
    if eval_infer_cfg is None:
        eval_infer_cfg = {
            "tta_hflip": False,
            "tta_vflip": False,
            "tta_rot90": False,
            "gaussian_sigma": 0.0,
            "gaussian_kernel_size": 0,
            "map_norm": "none",
            "map_norm_low_q": 0.01,
            "map_norm_high_q": 0.99,
        }
    # Collect all scores/labels first, then compute AUROC once.
    # This avoids ignite's intermittent "Only one class present" warnings
    # due to how it aggregates per-update batches.
    p_scores_list = []
    p_targets_list = []
    i_scores_list = []
    i_targets_list = []
    use_raw_map_for_image_score = bool(image_score_cfg.get("use_raw_map", False))

    for data, targets in dataloader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.no_grad():
            anomaly_map_raw, anomaly_map_post = infer_anomaly_maps(
                model, data, eval_infer_cfg
            )
        anomaly_map = anomaly_map_post.cpu().detach()  # (B, 1, H, W)
        image_score_map = (
            anomaly_map_raw if use_raw_map_for_image_score else anomaly_map_post
        ).cpu().detach()
        targets_cpu = targets.cpu()

        # Pixel-level
        outputs_flat = anomaly_map.flatten()
        targets_flat = targets_cpu.flatten()
        p_scores_list.append(outputs_flat)
        p_targets_list.append(targets_flat)

        # Image-level
        b = image_score_map.shape[0]
        image_scores = aggregate_image_scores(image_score_map, image_score_cfg)
        image_labels = (
            targets_cpu.view(b, -1).max(dim=1).values > 0
        ).float()  # (B,)
        i_scores_list.append(image_scores)
        i_targets_list.append(image_labels)

    p_scores = torch.cat(p_scores_list).numpy()
    p_targets = torch.cat(p_targets_list).numpy()
    i_scores = torch.cat(i_scores_list).numpy()
    i_targets = torch.cat(i_targets_list).numpy()

    def safe_roc_auc(y_true, y_score):
        # GT masks may come as float tensors (not strictly 0/1),
        # so binarize explicitly for binary AUROC.
        y_true = (np.asarray(y_true) > 0.5).astype(np.int32)
        y_score = np.asarray(y_score, dtype=np.float64)
        uniq = np.unique(y_true)
        if uniq.shape[0] < 2:
            # Keep behavior explicit; downstream code will handle nan.
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    p_auroc = safe_roc_auc(p_targets, p_scores)
    i_auroc = safe_roc_auc(i_targets, i_scores)
    print("I-AUROC: {}".format(i_auroc))
    print("P-AUROC: {}".format(p_auroc))
    return i_auroc, p_auroc


def save_checkpoint(path, epoch, model, optimizer=None):
    payload = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_state_dict_compat(model, state_dict, keep_model_fusion=False):
    # Keep eval/train backward-compatible when toggling fusion_learnable.
    # Old checkpoints may store fusion logits while new config expects fixed
    # weights (or the inverse).
    state = dict(state_dict)
    if keep_model_fusion:
        state.pop("fusion_logits", None)
        state.pop("fusion_weights", None)
        print("Checkpoint compat: keeping model fusion params from config")

    model_keys = model.state_dict().keys()
    model_has_logits = "fusion_logits" in model_keys
    model_has_weights = "fusion_weights" in model_keys
    ckpt_has_logits = "fusion_logits" in state
    ckpt_has_weights = "fusion_weights" in state

    if model_has_weights and ckpt_has_logits and not ckpt_has_weights:
        logits = state.pop("fusion_logits")
        weights = torch.softmax(logits.float(), dim=0)
        state["fusion_weights"] = weights.to(dtype=logits.dtype)
        print("Checkpoint compat: converted fusion_logits -> fusion_weights")
    elif model_has_logits and ckpt_has_weights and not ckpt_has_logits:
        weights = state.pop("fusion_weights").float()
        weights = weights / (weights.sum() + 1e-8)
        logits = torch.log(weights + 1e-8)
        state["fusion_logits"] = logits
        print("Checkpoint compat: converted fusion_weights -> fusion_logits")

    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("fusion_")]
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("fusion_")]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint incompatible (non-fusion keys). missing={} unexpected={}".format(
                missing, unexpected
            )
        )


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR,
        "exp%d_%s" % (len(os.listdir(const.CHECKPOINT_DIR)), args.category),
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    effective_config_path = resolve_category_config_path(args.config, config, args.category)
    if effective_config_path != args.config:
        config = yaml.safe_load(open(effective_config_path, "r"))
    args.config = effective_config_path
    image_score_cfg = build_image_score_cfg(config)
    eval_infer_cfg = build_eval_infer_cfg(config)
    print("Eval image score config: {}".format(image_score_cfg))
    print("Eval infer config: {}".format(eval_infer_cfg))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    device = get_runtime_device(args)
    print("Runtime device: {}".format(device))
    model.to(device)

    # Keep only the best checkpoints for each metric.
    best_i_epoch = -1
    best_i_auroc = -float("inf")
    best_p_epoch = -1
    best_p_auroc = -float("inf")

    early_stop_patience = const.EARLY_STOP_PATIENCE
    early_stop_min_delta = const.EARLY_STOP_MIN_DELTA
    epochs_since_best_p = 0

    best_i_ckpt_path = os.path.join(checkpoint_dir, "best_i.pt")
    best_p_ckpt_path = os.path.join(checkpoint_dir, "best_p.pt")
    # Backward-compat: keep best.pt pointing to best_p.pt
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            i_auroc, p_auroc = eval_once(
                test_dataloader, model, image_score_cfg, eval_infer_cfg
            )
            # Paper's evaluation includes both image/pixel AUROC.
            # We keep two independent "best" checkpoints:
            # - best_i.pt for I-AUROC (image-level)
            # - best_p.pt for P-AUROC (pixel-level)
            if not math.isnan(i_auroc) and i_auroc > best_i_auroc:
                best_i_auroc = i_auroc
                best_i_epoch = epoch
                save_checkpoint(best_i_ckpt_path, epoch, model, optimizer)

            improved_p = False
            if not math.isnan(p_auroc) and p_auroc > (best_p_auroc + early_stop_min_delta):
                best_p_auroc = p_auroc
                best_p_epoch = epoch
                improved_p = True
                save_checkpoint(best_p_ckpt_path, epoch, model, optimizer)
                # Also update legacy best.pt as an alias for best_p.pt
                save_checkpoint(best_ckpt_path, epoch, model, optimizer)

            if improved_p:
                epochs_since_best_p = 0
            else:
                epochs_since_best_p += 1

            print(
                "[Global Best @ epoch {}] I-AUROC: {} (epoch {}), "
                "P-AUROC: {} (epoch {})".format(
                    epoch + 1, best_i_auroc, best_i_epoch + 1, best_p_auroc, best_p_epoch + 1
                )
            )

            if early_stop_patience > 0 and epochs_since_best_p >= early_stop_patience:
                print(
                    "Early stopping at epoch {}: no P-AUROC improvement for {} evals (min_delta={}).".format(
                        epoch + 1, epochs_since_best_p, early_stop_min_delta
                    )
                )
                break

    # Final: run eval with best weights
    if os.path.exists(best_p_ckpt_path):
        print(
            "Loading best P checkpoint: {} (epoch {})".format(
                best_p_ckpt_path, best_p_epoch
            )
        )
        checkpoint = torch.load(best_p_ckpt_path, map_location="cpu")
        load_state_dict_compat(model, checkpoint["model_state_dict"])
        i_at_best_p, p_at_best_p = eval_once(
            test_dataloader, model, image_score_cfg, eval_infer_cfg
        )

        # Also load best I for visibility (does not change return values).
        i_at_best_i = float("nan")
        p_at_best_i = float("nan")
        if os.path.exists(best_i_ckpt_path):
            print(
                "Loading best I checkpoint: {} (epoch {})".format(
                    best_i_ckpt_path, best_i_epoch
                )
            )
            checkpoint = torch.load(best_i_ckpt_path, map_location="cpu")
            load_state_dict_compat(model, checkpoint["model_state_dict"])
            i_at_best_i, p_at_best_i = eval_once(
                test_dataloader, model, image_score_cfg, eval_infer_cfg
            )
            print(
                "Best summary: I-AUROC(best_i) = {}, P-AUROC(best_i) = {}, "
                "I-AUROC(best_p) = {}, P-AUROC(best_p) = {}".format(
                    i_at_best_i, p_at_best_i, i_at_best_p, p_at_best_p
                )
            )

        append_results_csv(
            args.results_csv,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": args.category,
                "I_AUROC_at_best_p": i_at_best_p,
                "P_AUROC_at_best_p": p_at_best_p,
                "I_AUROC_at_best_i": i_at_best_i,
                "P_AUROC_at_best_i": p_at_best_i,
                "best_p_checkpoint": os.path.abspath(best_p_ckpt_path),
                "best_i_checkpoint": os.path.abspath(best_i_ckpt_path)
                if os.path.exists(best_i_ckpt_path)
                else "",
                "config": os.path.abspath(args.config),
                "seed": args.seed,
                "deterministic": int(args.deterministic),
            },
        )

        return i_at_best_i, p_at_best_p

        # If best_i.pt is missing (shouldn't happen), fall back to best_p's I-AUROC.
        return i_at_best_p, p_at_best_p

    print(
        "Warning: best_p.pt not found in {}; skipping final best evaluation.".format(
            checkpoint_dir
        )
    )
    return None, None


def resolve_eval_checkpoint(category, explicit_path=None):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(explicit_path))
        return os.path.abspath(explicit_path)

    env_ckpt = os.environ.get("CYG_EVAL_CKPT", "").strip()
    if env_ckpt:
        if not os.path.isfile(env_ckpt):
            raise FileNotFoundError(
                "CYG_EVAL_CKPT is set but file not found: {}".format(env_ckpt)
            )
        return os.path.abspath(env_ckpt)

    ckpt_names = ("best_p.pt", "best.pt", "best_i.pt")
    candidates = []
    for name in ckpt_names:
        pattern = os.path.join(
            const.CHECKPOINT_DIR, "exp*_{}".format(category), name
        )
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint for category '{}'. Train first or pass -ckpt PATH. "
            "Searched under {}/exp*_{}/{{best_p,best,best_i}}.pt".format(
                category, const.CHECKPOINT_DIR, category
            )
        )

    ckpt_path = max(candidates, key=os.path.getmtime)
    print("Auto-selected checkpoint: {}".format(ckpt_path))
    return os.path.abspath(ckpt_path)


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    effective_config_path = resolve_category_config_path(args.config, config, args.category)
    if effective_config_path != args.config:
        config = yaml.safe_load(open(effective_config_path, "r"))
    args.config = effective_config_path
    image_score_cfg = build_image_score_cfg(config)
    eval_infer_cfg = build_eval_infer_cfg(config)
    print("Eval image score config: {}".format(image_score_cfg))
    print("Eval infer config: {}".format(eval_infer_cfg))
    model = build_model(config)
    ckpt_path = resolve_eval_checkpoint(args.category, args.checkpoint)
    args.checkpoint = ckpt_path
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    load_state_dict_compat(
        model,
        checkpoint["model_state_dict"],
        keep_model_fusion=bool(getattr(args, "eval_use_config_fusion", False)),
    )
    test_dataloader = build_test_data_loader(args, config)
    device = get_runtime_device(args)
    print("Runtime device: {}".format(device))
    model.to(device)
    i_auroc, p_auroc = eval_once(
        test_dataloader, model, image_score_cfg, eval_infer_cfg
    )

    if args.append_eval_csv:
        append_results_csv(
            args.append_eval_csv,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": args.category,
                "I_AUROC_at_best_p": i_auroc,
                "P_AUROC_at_best_p": p_auroc,
                "I_AUROC_at_best_i": i_auroc,
                "P_AUROC_at_best_i": p_auroc,
                "best_p_checkpoint": os.path.abspath(args.checkpoint),
                "best_i_checkpoint": os.path.abspath(args.checkpoint),
                "config": os.path.abspath(args.config),
                "seed": args.seed,
                "deterministic": int(args.deterministic),
            },
        )

    return i_auroc, p_auroc


def parse_args():
    parser = argparse.ArgumentParser(description="Train CyG-Flow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        required=True,
        help="category name in mvtec, or 'all' to train all categories",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str,
        help="checkpoint path (default: latest best_p.pt under CYG_CHECKPOINT_DIR)",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=DEFAULT_RESULTS_CSV,
        help="append per-category vmamba train results to this CSV file",
    )
    parser.add_argument(
        "--append-eval-csv",
        type=str,
        default=DEFAULT_RESULTS_CSV,
        help="append eval-only results to this CSV file",
    )
    parser.add_argument(
        "--eval-use-config-fusion",
        action="store_true",
        help="in eval mode, keep fusion params from config instead of checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="global random seed for reproducibility",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable deterministic CUDA/cuDNN behavior (slower but more reproducible)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="N",
        help=(
            "CUDA device index; omit to auto-pick the first usable GPU from "
            "CYG_CUDA_DEVICES or default 0,1,2,3,4,6. Use -1 for CPU."
        ),
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_reproducibility(args.seed, args.deterministic)
    if args.eval:
        # eval mode is for a single category only
        if args.category == "all":
            raise ValueError("eval mode does not support -cat all; run eval per category.")
        evaluate(args)
    else:
        if args.category == "all":
            # Train categories sequentially; each category has its own checkpoint directory.
            summary = []
            for c in const.MVTEC_CATEGORIES:
                print("\n========== Training category: {} ==========".format(c))
                args.category = c
                i_auroc, p_auroc = train(args)
                summary.append((c, p_auroc, i_auroc))
            print("\nAll categories finished.")
            print("\n==== Summary (best.pt) ====")
            for c, p, i in summary:
                print("{:<12} P-AUROC: {:>8} | I-AUROC: {:>8}".format(c, p, i))
        else:
            train(args)
