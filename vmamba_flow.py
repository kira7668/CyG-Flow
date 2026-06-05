"""CyG-Flow: VMamba (vssm_small) + normalizing flow (Glow 1x1 + stable_tanh RealNVP + CRF-B subnets)."""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const
import flow as _base
from FrEIA.modules import InvertibleModule

_SELECTIVE_SCAN_TORCH_PATCH_DONE = False


def _parse_gpu_arg_from_argv():
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            try:
                return int(argv[i + 1].strip())
            except ValueError:
                return None
        if a.startswith("--gpu="):
            try:
                return int(a.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _ensure_fvcore_for_vmamba():
    try:
        import fvcore.nn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "VMamba (vssm_small) requires fvcore. Install with: pip install fvcore"
        ) from e


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _should_use_torch_selective_scan_fallback():
    if _env_truthy("CYG_VM_FORCE_TORCH_SELECTIVE_SCAN"):
        return True, "CYG_VM_FORCE_TORCH_SELECTIVE_SCAN"
    if _env_truthy("CYG_VM_DISABLE_AUTO_TORCH_SELECTIVE_SCAN"):
        return False, None
    if not torch.cuda.is_available():
        return False, None

    forced_gpu = _parse_gpu_arg_from_argv()
    if forced_gpu is not None and forced_gpu < 0:
        return False, None

    if forced_gpu is not None:
        n = torch.cuda.device_count()
        if forced_gpu >= n:
            return False, None
        major, minor = torch.cuda.get_device_capability(forced_gpu)
        if (major, minor) < (8, 9):
            return True, "auto (--gpu {} is sm_{}.{})".format(
                forced_gpu, major, minor
            )
        return False, None

    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if (major, minor) < (8, 9):
            return True, "auto (visible device {} is sm_{}.{})".format(
                i, major, minor
            )
    return False, None


def _install_selective_scan_torch_workaround():
    global _SELECTIVE_SCAN_TORCH_PATCH_DONE
    if _SELECTIVE_SCAN_TORCH_PATCH_DONE:
        return
    use_torch, reason = _should_use_torch_selective_scan_fallback()
    if not use_torch:
        return

    import VMamba.classification.models.csms6s as csms6s

    _orig = csms6s.selective_scan_fn

    def _wrapped(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        delta_bias=None,
        delta_softplus=True,
        oflex=True,
        backend=None,
    ):
        b = backend
        if b is None or b == "oflex":
            b = "torch"
        return _orig(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            delta_softplus,
            oflex,
            b,
        )

    csms6s.selective_scan_fn = _wrapped
    print(
        "[vmamba_flow] selective_scan -> torch (slower; reason: {}). "
        "Recompile VMamba selective_scan for your sm to use CUDA.".format(reason),
        flush=True,
    )
    for _mod_name, _mod in list(sys.modules.items()):
        if _mod is None:
            continue
        if _mod_name.endswith("models.vmamba") and hasattr(_mod, "selective_scan_fn"):
            _mod.selective_scan_fn = _wrapped
    _SELECTIVE_SCAN_TORCH_PATCH_DONE = True


class _LogAbsDetWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W: torch.Tensor):
        _, logabsdet = torch.linalg.slogdet(W.detach().cpu().double())
        ctx.save_for_backward(W)
        return logabsdet.float().to(device=W.device, dtype=W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (W,) = ctx.saved_tensors
        go = grad_output.detach().cpu().double().reshape(())
        with torch.no_grad():
            wi_t = torch.linalg.inv(W.cpu().double()).t()
        grad_W = (go * wi_t).float().to(device=W.device, dtype=W.dtype)
        return grad_W


class _InvWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W: torch.Tensor):
        wi = torch.linalg.inv(W.detach().cpu().double()).to(
            device=W.device, dtype=W.dtype
        )
        ctx.save_for_backward(W)
        return wi

    @staticmethod
    def backward(ctx, grad_Wi: torch.Tensor):
        (W,) = ctx.saved_tensors
        with torch.no_grad():
            wi = torch.linalg.inv(W.cpu().double())
            g = grad_Wi.cpu().double()
            grad_W = -(wi.t().mm(g).mm(wi.t())).float().to(
                device=W.device, dtype=W.dtype
            )
        return grad_W


class Invertible1x1ConvGlow(InvertibleModule):
    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)
        c = int(dims_in[0][0])
        self.c = c
        w = torch.randn(c, c)
        w = torch.linalg.qr(w, mode="reduced")[0]
        self.weight = nn.Parameter(w)

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} expects a single input")
        return input_dims

    def forward(self, x, c=[], rev=False, jac=True):
        _ = c
        xt = x[0]
        b, ch, h, w_sp = xt.shape
        if ch != self.c:
            raise ValueError(
                f"Invertible1x1ConvGlow: expected C={self.c}, got {ch}"
            )
        W = self.weight
        k = self.c
        if not rev:
            z = F.conv2d(xt, W.view(k, k, 1, 1))
            if jac:
                logabsdet = _LogAbsDetWeight.apply(W)
                j = (h * w_sp) * logabsdet
                j = j.expand(b).to(dtype=xt.dtype, device=xt.device)
            else:
                j = torch.zeros(b, device=xt.device, dtype=xt.dtype)
            return [z], j
        Wi = _InvWeight.apply(W)
        z = F.conv2d(xt, Wi.view(k, k, 1, 1))
        if jac:
            logabsdet = _LogAbsDetWeight.apply(W)
            j = -(h * w_sp) * logabsdet
            j = j.expand(b).to(dtype=xt.dtype, device=xt.device)
        else:
            j = torch.zeros(b, device=xt.device, dtype=xt.dtype)
        return [z], j


def nf_cyg_flow_vmamba(
    input_chw,
    conv3x3_only,
    hidden_ratio,
    flow_steps,
    clamp=2.0,
    flow_activation="relu",
    flow_subnet_type="standard",
    flow_kernel_alternating=True,
    flow_crf_b_alternating_period=1,
    flow_coupling_type="standard",
    flow_coupling_eps=0.02,
    flow_coupling_alpha=1.0,
    flow_coupling_gamma=0.1,
    inv1x1_every_n: int = 2,
):
    """Build the CyG normalizing-flow stack used by ``CyGFlow``.

    Cyclic CRF-B receptive fields: ``1x1 -> 3x3 -> 5x5``. The ``5x5`` stage is
    implemented as two stacked ``3x3`` convolutions (same effective RF).
    """
    coupling_type = str(flow_coupling_type).lower()
    if coupling_type in ["standard", "aio", "allinone", "all_in_one", "default"]:
        coupling_type = "standard"
    elif coupling_type in [
        "stable",
        "stable_tanh",
        "stable_tanh_affine",
        "tanh_affine",
        "safe_affine",
    ]:
        coupling_type = "stable_tanh_affine"
    elif coupling_type in [
        "stable_tanh_realnvp",
        "stable_realnvp",
        "tanh_realnvp",
        "safe_realnvp",
    ]:
        coupling_type = "stable_tanh_realnvp"
    else:
        raise ValueError(
            "flow_coupling_type must be one of "
            "['standard', 'stable_tanh_affine', 'stable_tanh_realnvp']"
        )

    block_cls = _base.Fm.AllInOneBlock
    if coupling_type == "stable_tanh_affine":
        block_cls = _base.StableTanhAffineCouplingBlock
    elif coupling_type == "stable_tanh_realnvp":
        block_cls = _base.StableTanhRealNVPCouplingBlock

    nodes = _base.Ff.SequenceINN(*input_chw)
    # 1x1 -> 3x3 -> 5x5 cycle; 5x5 is two stacked 3x3 (kernel plan [1, 3, 3]).
    base_kernel_plan = [1, 3, 3]
    if flow_steps <= 0:
        flow_steps = 1
    kernel_plan = [base_kernel_plan[i % len(base_kernel_plan)] for i in range(flow_steps)]

    inv_n = int(inv1x1_every_n)
    for i, kernel_size in enumerate(kernel_plan):
        if inv_n > 0 and i > 0 and (i % inv_n == 0):
            nodes.append(Invertible1x1ConvGlow)

        k = int(kernel_size)
        subnet_constructor = _base.subnet_conv_func(
            k,
            hidden_ratio,
            activation=flow_activation,
            subnet_type=flow_subnet_type,
        )
        block_kwargs = dict(
            subnet_constructor=subnet_constructor,
            affine_clamping=clamp,
            permute_soft=False,
        )
        if coupling_type in ["stable_tanh_affine", "stable_tanh_realnvp"]:
            block_kwargs.update(
                stable_eps=flow_coupling_eps,
                shift_alpha=flow_coupling_alpha,
            )
        if coupling_type == "stable_tanh_realnvp":
            block_kwargs.update(scale_gamma=flow_coupling_gamma)
        nodes.append(block_cls, **block_kwargs)
    return nodes


class CyGFlow(_base.CyGFlow):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
        vssm_ckpt=None,
        vssm_out_indices=None,
        use_saf=False,
        fusion_weights=None,
        fusion_learnable=False,
        fusion_init=None,
        fusion_type="weighted",
        fusion_pre_norm="none",
        fusion_pre_norm_clip=0.0,
        fusion_pre_norm_eps=1e-6,
        fusion_pre_norm_power=1.0,
        fusion_pre_norm_low_q=0.01,
        fusion_pre_norm_high_q=0.99,
        flow_activation="relu",
        flow_subnet_type="standard",
        flow_kernel_alternating=True,
        flow_crf_b_alternating_period=1,
        flow_coupling_type="standard",
        flow_coupling_eps=0.02,
        flow_coupling_alpha=1.0,
        flow_coupling_gamma=0.1,
        inv1x1_every_n: int = 2,
    ):
        if str(backbone_name).strip() != const.BACKBONE_VSSM_SMALL:
            raise ValueError(
                "CyG-Flow only supports backbone_name=vssm_small, got {!r}".format(
                    backbone_name
                )
            )

        _ensure_fvcore_for_vmamba()
        _install_selective_scan_torch_workaround()

        _base.nn.Module.__init__(self)
        self.input_size = input_size

        vmamba_models_dir = os.path.join(
            os.path.dirname(__file__), "VMamba", "classification", "models"
        )
        if vmamba_models_dir not in sys.path:
            sys.path.insert(0, vmamba_models_dir)
        from VMamba.classification.models.vmamba import Backbone_VSSM

        if vssm_ckpt is None:
            vssm_ckpt = os.path.join(
                os.path.dirname(__file__),
                "vim_small_midclstok",
                "vssm_small_0229_ckpt_epoch_222.pth",
            )

        if vssm_out_indices is None:
            vssm_out_indices = (0, 1, 2)
        vssm_out_indices = tuple(int(i) for i in vssm_out_indices)
        if (
            len(vssm_out_indices) < 1
            or len(vssm_out_indices) > 4
            or any(i < 0 or i > 3 for i in vssm_out_indices)
        ):
            raise ValueError(
                "vssm_out_indices must contain 1~4 stage indices in [0,1,2,3]"
            )
        if len(set(vssm_out_indices)) != len(vssm_out_indices):
            raise ValueError("vssm_out_indices must not contain duplicated stages")

        stage_channels = {0: 96, 1: 192, 2: 384, 3: 768}
        stage_scales = {0: 4, 1: 8, 2: 16, 3: 32}

        self.feature_extractor = Backbone_VSSM(
            out_indices=vssm_out_indices,
            pretrained=vssm_ckpt,
            norm_layer="ln2d",
            depths=[2, 2, 15, 2],
            dims=96,
            drop_path_rate=0.3,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
        )
        channels = [stage_channels[i] for i in vssm_out_indices]
        scales = [stage_scales[i] for i in vssm_out_indices]
        self.norms = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    elementwise_affine=True,
                )
            )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_cyg_flow_vmamba(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                    flow_activation=flow_activation,
                    flow_subnet_type=flow_subnet_type,
                    flow_kernel_alternating=flow_kernel_alternating,
                    flow_crf_b_alternating_period=flow_crf_b_alternating_period,
                    flow_coupling_type=flow_coupling_type,
                    flow_coupling_eps=flow_coupling_eps,
                    flow_coupling_alpha=flow_coupling_alpha,
                    flow_coupling_gamma=flow_coupling_gamma,
                    inv1x1_every_n=inv1x1_every_n,
                )
            )

        self.use_saf = use_saf
        if self.use_saf:
            self.saf_blocks = nn.ModuleList([_base.SAF(c) for c in channels])
        else:
            self.saf_blocks = None
        self.channels = channels
        self.scales = scales

        self.fusion_type = fusion_type
        self.fusion_pre_norm = str(fusion_pre_norm).lower()
        self.fusion_pre_norm_clip = float(fusion_pre_norm_clip)
        self.fusion_pre_norm_eps = float(fusion_pre_norm_eps)
        self.fusion_pre_norm_power = float(fusion_pre_norm_power)
        self.fusion_pre_norm_low_q = float(fusion_pre_norm_low_q)
        self.fusion_pre_norm_high_q = float(fusion_pre_norm_high_q)
        if self.fusion_pre_norm not in ["none", "minmax", "zscore", "robust"]:
            raise ValueError(
                "fusion_pre_norm must be one of ['none', 'minmax', 'zscore', 'robust']"
            )
        if self.fusion_pre_norm_power <= 0:
            raise ValueError("fusion_pre_norm_power must be > 0")
        if not (0.0 <= self.fusion_pre_norm_low_q < self.fusion_pre_norm_high_q <= 1.0):
            raise ValueError("fusion_pre_norm_low_q/high_q must satisfy 0<=low<high<=1")
        self.fusion_learnable = bool(fusion_learnable)
        self.fusion_logits = None
        if self.fusion_learnable:
            n = len(channels)
            if fusion_init is not None:
                if len(fusion_init) != n:
                    raise ValueError(
                        f"fusion_init length {len(fusion_init)} must match scales {n}"
                    )
                init = _base.torch.tensor(fusion_init, dtype=_base.torch.float32)
            else:
                init = _base.torch.ones(n, dtype=_base.torch.float32)
            init = init / init.sum()
            self.fusion_logits = nn.Parameter(init.log())
        else:
            if fusion_weights is None:
                fusion_weights = [1.0 / len(channels)] * len(channels)
            if len(fusion_weights) != len(channels):
                raise ValueError(
                    f"fusion_weights length {len(fusion_weights)} must match scales {len(channels)}"
                )
            fw = _base.torch.tensor(fusion_weights, dtype=_base.torch.float32)
            fw = fw / fw.sum()
            self.register_buffer("fusion_weights", fw)

    forward = _base.CyGFlow.forward


for _name in dir(_base):
    if _name.startswith("__") or _name == "CyGFlow":
        continue
    globals()[_name] = getattr(_base, _name)
