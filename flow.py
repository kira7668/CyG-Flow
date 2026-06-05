import FrEIA.framework as Ff
import FrEIA.modules as Fm
import os
import sys
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const

CRF_BLOCK_DIR = os.path.join(os.path.dirname(__file__), "CRFBlock")
if CRF_BLOCK_DIR not in sys.path:
    sys.path.insert(0, CRF_BLOCK_DIR)

from models.crf_b_blocks import BasicCRF_B, BasicCRF_B_a


class SAF(nn.Module):
    """Self-Alignment Function: lightweight per-scale residual MLP."""

    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x + self.mlp(x)


class CRF_BDualBranchSubnet(nn.Module):
    """Legacy dual-branch CRF-B subnet used by older checkpoints."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer):
        super().__init__()
        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 1)

        self.branch_1x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
            ),
            act_layer(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.branch_3x1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
            act_layer(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.residual = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        merged = self.branch_1x3(x) + self.branch_3x1(x)
        return self.residual(merged)


class CRF_BLiteSubnet(nn.Module):
    """CRF-B-Lite style subnet with asymmetric and dilated receptive fields."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer, kernel_size=3):
        super().__init__()
        k = int(kernel_size)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        pad = k // 2

        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 3)
        branch_channels = max(hidden_channels // 3, 1)

        self.branch_identity = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
        )
        self.branch_asym_13_31 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(1, k),
                padding=(0, pad),
                bias=False,
            ),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(k, 1),
                padding=(pad, 0),
                bias=False,
            ),
        )
        self.branch_asym_31_13 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(k, 1),
                padding=(pad, 0),
                bias=False,
            ),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(1, k),
                padding=(0, pad),
                bias=False,
            ),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
        )

        self.fuse = nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.out_act = act_layer()

    def forward(self, x):
        merged = torch.cat(
            [
                self.branch_identity(x),
                self.branch_asym_13_31(x),
                self.branch_asym_31_13(x),
                self.branch_dilated(x),
            ],
            dim=1,
        )
        out = self.fuse(merged)
        out = out + self.shortcut(x)
        return self.out_act(out)


class CRF_BMultiScaleSubnet(nn.Module):
    """CRF-B-style multi-scale subnet for flow blocks without 3x3/1x1 alternation."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer, kernel_size=3):
        super().__init__()
        k = int(kernel_size)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 4)
        branch_channels = max(hidden_channels // 4, 1)

        self.branch_local = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            act_layer(),
        )
        self.branch_mid = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            act_layer(),
        )
        self.branch_large = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=False,
            ),
            act_layer(),
        )
        self.branch_context = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=k,
                padding=k // 2,
                groups=branch_channels,
                bias=False,
            ),
            act_layer(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False),
        )

        self.fuse = nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.out_act = act_layer()

    def forward(self, x):
        merged = torch.cat(
            [
                self.branch_local(x),
                self.branch_mid(x),
                self.branch_large(x),
                self.branch_context(x),
            ],
            dim=1,
        )
        out = self.fuse(merged)
        out = out + self.shortcut(x)
        return self.out_act(out)


class CRF_BNetBasicConv(nn.Module):
    """BasicConv adapted for flow subnet use."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = act_layer() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CRF_BNetSubnet(nn.Module):
    """CRF-BNet BasicCRF_B block (crf_b_blocks.py) as flow subnet."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.block = BasicCRF_B(
            in_channels,
            out_channels,
            stride=1,
            scale=self.scale,
            visual=1,
        )

    def forward(self, x):
        return self.block(x)


class CRF_BNetSSubnet(nn.Module):
    """CRF-BNet-S style block based on BasicCRF_B_a (crf_b_blocks.py)."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.block = BasicCRF_B_a(
            in_channels,
            out_channels,
            stride=1,
            scale=self.scale,
        )

    def forward(self, x):
        return self.block(x)


class GatedPointwiseSubnet(nn.Module):
    """1x1 gated pointwise subnet used as a drop-in replacement for 1x1 flow blocks."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer):
        super().__init__()
        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 1)

        self.gate_proj = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = act_layer()

    def forward(self, x):
        value, gate = torch.chunk(self.gate_proj(x), 2, dim=1)
        y = self.act(value) * torch.sigmoid(gate)
        return self.out_proj(y) + self.shortcut(x)


class ResidualConvSubnet(nn.Module):
    """Residual 3x3/1x1 conv subnet as a non-CRF-B alternative."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer, kernel_size=3):
        super().__init__()
        k = int(kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        pad = k // 2

        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 1)

        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=k,
            padding=pad,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=k,
            padding=pad,
            bias=False,
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = act_layer()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return y + self.shortcut(x)


class WideGatedPointwiseSubnet(nn.Module):
    """Wider gated 1x1 subnet to preserve capacity in alternating-flow setups."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer):
        super().__init__()
        hidden_channels = int(in_channels * hidden_ratio * 2.0)
        hidden_channels = max(hidden_channels, 1)

        self.in_proj = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.mid_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = act_layer()

    def forward(self, x):
        value, gate = torch.chunk(self.in_proj(x), 2, dim=1)
        y = self.act(value) * torch.sigmoid(gate)
        y = self.act(self.mid_proj(y))
        return self.out_proj(y) + self.shortcut(x)


class ConvNeXtAltSubnet(nn.Module):
    """Depthwise-pointwise residual subnet as another non-CRF-B alternative."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer, kernel_size=3):
        super().__init__()
        k = int(kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        pad = k // 2

        # Use a wider expansion at 3x3 steps to preserve representational capacity.
        expansion = 8.0 if k > 1 else 4.0
        hidden_channels = int(in_channels * hidden_ratio * expansion)
        hidden_channels = max(hidden_channels, 1)

        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=k,
            padding=pad,
            groups=in_channels,
            bias=False,
        )
        self.pw1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = act_layer()

    def forward(self, x):
        y = self.dw(x)
        y = self.act(self.pw1(y))
        y = self.pw2(y)
        return y + self.shortcut(x)


class DualPathStableSubnet(nn.Module):
    """Stable dual-path residual subnet mixing local and depthwise routes."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer, kernel_size=3):
        super().__init__()
        k = int(kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        pad = k // 2

        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 1)

        self.local = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=k,
                padding=pad,
                bias=False,
            ),
            act_layer(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=k,
                padding=pad,
                bias=False,
            ),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=k,
                padding=pad,
                groups=in_channels,
                bias=False,
            ),
            act_layer(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        local = self.local(x)
        context = self.context(x)
        mix = torch.sigmoid(self.mix_logit)
        return mix * local + (1.0 - mix) * context + self.shortcut(x)


class ResidualPointwiseSubnet(nn.Module):
    """Residual 1x1 subnet to improve stability for alternating flow blocks."""

    def __init__(self, in_channels, out_channels, hidden_ratio, act_layer):
        super().__init__()
        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 1)

        self.pw1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = act_layer()

    def forward(self, x):
        y = self.act(self.pw1(x))
        y = self.pw2(y)
        return y + self.shortcut(x)


class StableTanhAffineCouplingBlock(Fm.AllInOneBlock):
    """A numerically stable affine coupling variant for s/t parameterization.

    It keeps AllInOneBlock's structure but changes affine transform to:
      log_e = clamp * tanh(s) + eps * s
      t = alpha * tanh(t)
      y = exp(log_e) * (x + t)
    """

    def __init__(self, *args, stable_eps=0.02, shift_alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.stable_eps = max(0.0, float(stable_eps))
        self.shift_alpha = max(0.0, float(shift_alpha))

    def _affine(self, x, a, rev=False):
        # Match the original block's coefficient scaling for stable initialization.
        a = a * 0.1
        ch = x.shape[1]

        s_raw = a[:, :ch]
        log_e = self.clamp * torch.tanh(s_raw) + self.stable_eps * s_raw
        # Guard against overflow in exp(log_e) on out-of-distribution checkpoints.
        log_e = torch.clamp(log_e, min=-20.0, max=20.0)
        if self.GIN:
            log_e = log_e - torch.mean(log_e, dim=self.sum_dims, keepdim=True)

        t = self.shift_alpha * torch.tanh(a[:, ch:])

        if not rev:
            return (
                torch.exp(log_e) * (x + t),
                torch.sum(log_e, dim=self.sum_dims),
            )

        return (
            torch.exp(-log_e) * x - t,
            -torch.sum(log_e, dim=self.sum_dims),
        )


class StableTanhRealNVPCouplingBlock(Fm.AllInOneBlock):
    """Stable RealNVP-style affine coupling.

        Compared with StableTanhAffineCouplingBlock, this variant uses
        y = exp(log_e) * x + t (instead of exp(log_e) * (x + t)).
        To keep compatibility with pretrained standard checkpoints, it keeps
        the original affine core and applies residual stabilizers:
            log_e = clamp * tanh(s) + gamma * eps * s
            t = t_raw + alpha * tanh(t_raw)
    """

    def __init__(
        self,
        *args,
        stable_eps=0.02,
        shift_alpha=1.0,
        scale_gamma=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stable_eps = max(0.0, float(stable_eps))
        self.shift_alpha = max(0.0, float(shift_alpha))
        self.scale_gamma = max(0.0, float(scale_gamma))

    def _affine(self, x, a, rev=False):
        a = a * 0.1
        ch = x.shape[1]

        s_raw = a[:, :ch]
        sub_jac = self.clamp * torch.tanh(s_raw)
        log_e = sub_jac + self.scale_gamma * (self.stable_eps * s_raw)
        log_e = torch.clamp(log_e, min=-20.0, max=20.0)
        if self.GIN:
            log_e = log_e - torch.mean(log_e, dim=self.sum_dims, keepdim=True)

        t_raw = a[:, ch:]
        t = t_raw + self.shift_alpha * torch.tanh(t_raw)

        if not rev:
            return (
                torch.exp(log_e) * x + t,
                torch.sum(log_e, dim=self.sum_dims),
            )

        return (
            torch.exp(-log_e) * (x - t),
            -torch.sum(log_e, dim=self.sum_dims),
        )


def subnet_conv_func(
    kernel_size,
    hidden_ratio,
    activation="relu",
    subnet_type="standard",
):
    activation = str(activation).lower()
    if activation == "relu":
        act_layer = nn.ReLU
    elif activation == "silu":
        act_layer = nn.SiLU
    elif activation == "gelu":
        act_layer = nn.GELU
    else:
        raise ValueError("flow_activation must be one of ['relu', 'silu', 'gelu']")

    subnet_type = str(subnet_type).lower()
    _legacy_cfb_subnet_aliases = {
        "cfb": "crf-b",
        "cfb_dual_branch": "crf-b_dual",
        "cfbnet": "crf-bnet",
        "cfbnets": "crf-bnets",
        "cfbnet_s": "crf-bnets",
        "cfb_s": "crf-bnets",
        "cfb_mix_alt": "crf-b_mix_alt",
        "cfb_mix": "crf-b_mix_alt",
        "cfbnet_cfbnets_alt": "crf-b_mix_alt",
        "mix_cfb": "crf-b_mix_alt",
        "cfb_ms": "crf-b_ms",
        "cfb_multiscale": "crf-b_ms",
        "cfb_multi_scale": "crf-b_ms",
        "cfb_plus": "crf-b_ms",
        "cfb_hybrid": "hybrid_alt",
        "non_cfb": "residual_alt",
    }
    subnet_type = _legacy_cfb_subnet_aliases.get(subnet_type, subnet_type)
    if subnet_type in ["crf-b", "crf-b_dual_branch"]:
        subnet_type = "crf-b_dual"
    if subnet_type in ["crf-bnet"]:
        subnet_type = "crf-bnet"
    if subnet_type in ["crf-bnets", "crf-bnet_s", "crf-b_s"]:
        subnet_type = "crf-bnets"
    if subnet_type in ["crf-b_mix_alt", "crf-b_mix", "crf-bnet_crf-bnets_alt", "mix_crf-b"]:
        subnet_type = "crf-b_mix_alt"
    if subnet_type in ["crf-b_ms", "crf-b_multiscale", "crf-b_multi_scale", "crf-b_plus"]:
        subnet_type = "crf-b_ms"
    if subnet_type in ["alternating", "hybrid", "crf-b_hybrid"]:
        subnet_type = "hybrid_alt"
    if subnet_type in ["residual", "res_alt", "residual_hybrid", "non_crf-b", "alt_res"]:
        subnet_type = "residual_alt"
    if subnet_type in ["residual_stable", "res_stable", "stable_residual"]:
        subnet_type = "residual_stable_alt"
    if subnet_type in ["convnext", "next_alt", "cnx_alt"]:
        subnet_type = "convnext_alt"
    if subnet_type in ["dualpath", "mix_alt", "stable_alt", "alt_dualpath", "dp_alt"]:
        subnet_type = "dualpath_alt"
    if subnet_type not in [
        "standard",
        "crf-bnet",
        "crf-bnets",
        "crf-b_mix_alt",
        "crf-b_dual",
        "crf-b_lite",
        "crf-b_ms",
        "hybrid_alt",
        "residual_alt",
        "residual_stable_alt",
        "convnext_alt",
        "dualpath_alt",
    ]:
        raise ValueError(
            "flow_subnet_type must be one of ['standard', 'crf-bnet', 'crf-bnets', 'crf-b_mix_alt', 'crf-b_dual', 'crf-b_lite', 'crf-b_ms', 'hybrid_alt', 'residual_alt', 'residual_stable_alt', 'convnext_alt', 'dualpath_alt']"
        )

    def subnet_conv(in_channels, out_channels):
        if subnet_type == "crf-b_mix_alt":
            raise RuntimeError(
                "crf-b_mix_alt should be resolved in nf_cyg_flow before subnet construction"
            )

        if subnet_type == "crf-bnet":
            return CRF_BNetSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
            )

        if subnet_type == "crf-bnets":
            return CRF_BNetSSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
            )

        if subnet_type == "crf-b_dual":
            return CRF_BDualBranchSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
            )

        if subnet_type == "crf-b_lite":
            return CRF_BLiteSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
                kernel_size=kernel_size,
            )

        if subnet_type == "crf-b_ms":
            return CRF_BMultiScaleSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
                kernel_size=kernel_size,
            )

        if subnet_type == "hybrid_alt":
            if int(kernel_size) == 1:
                return GatedPointwiseSubnet(
                    in_channels,
                    out_channels,
                    hidden_ratio=hidden_ratio,
                    act_layer=act_layer,
                )
            return CRF_BLiteSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
                kernel_size=kernel_size,
            )

        if subnet_type == "residual_alt":
            if int(kernel_size) == 1:
                return WideGatedPointwiseSubnet(
                    in_channels,
                    out_channels,
                    hidden_ratio=hidden_ratio,
                    act_layer=act_layer,
                )
            return ResidualConvSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
                kernel_size=kernel_size,
            )

        if subnet_type == "convnext_alt":
            return ConvNeXtAltSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
                kernel_size=kernel_size,
            )

        if subnet_type == "residual_stable_alt":
            if int(kernel_size) == 1:
                return ResidualPointwiseSubnet(
                    in_channels,
                    out_channels,
                    hidden_ratio=hidden_ratio,
                    act_layer=act_layer,
                )
            return ResidualConvSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
                kernel_size=kernel_size,
            )

        if subnet_type == "dualpath_alt":
            if int(kernel_size) == 1:
                return ResidualPointwiseSubnet(
                    in_channels,
                    out_channels,
                    hidden_ratio=hidden_ratio,
                    act_layer=act_layer,
                )
            return DualPathStableSubnet(
                in_channels,
                out_channels,
                hidden_ratio=hidden_ratio,
                act_layer=act_layer,
                kernel_size=kernel_size,
            )

        hidden_channels = int(in_channels * hidden_ratio)
        hidden_channels = max(hidden_channels, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            act_layer(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_cyg_flow(
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
):
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
            "flow_coupling_type must be one of ['standard', 'stable_tanh_affine', 'stable_tanh_realnvp']"
        )

    block_cls = Fm.AllInOneBlock
    if coupling_type == "stable_tanh_affine":
        block_cls = StableTanhAffineCouplingBlock
    elif coupling_type == "stable_tanh_realnvp":
        block_cls = StableTanhRealNVPCouplingBlock

    nodes = Ff.SequenceINN(*input_chw)
    use_alternating_kernels = bool(flow_kernel_alternating) and (not conv3x3_only)
    crf_b_period = max(int(flow_crf_b_alternating_period), 1)
    for i in range(flow_steps):
        subnet_type = flow_subnet_type
        if use_alternating_kernels and i % 2 == 1:
            kernel_size = 1
        else:
            kernel_size = 3
        if str(flow_subnet_type).lower() in [
            "crf-b_mix_alt",
            "crf-b_mix",
            "crf-bnet_crf-bnets_alt",
            "mix_crf-b",
            "cfb_mix_alt",
            "cfb_mix",
            "cfbnet_cfbnets_alt",
            "mix_cfb",
        ]:
            # Alternate between full CRF-BNet and CRF-BNet-S blocks by period.
            phase = (i // crf_b_period) % 2
            subnet_type = "crf-bnet" if phase == 0 else "crf-bnets"
        block_kwargs = dict(
            subnet_constructor=subnet_conv_func(
                kernel_size,
                hidden_ratio,
                activation=flow_activation,
                subnet_type=subnet_type,
            ),
            affine_clamping=clamp,
            permute_soft=False,
        )
        if coupling_type in ["stable_tanh_affine", "stable_tanh_realnvp"]:
            block_kwargs.update(
                stable_eps=flow_coupling_eps,
                shift_alpha=flow_coupling_alpha,
            )
        if coupling_type == "stable_tanh_realnvp":
            block_kwargs.update(
                scale_gamma=flow_coupling_gamma,
            )
        nodes.append(
            block_cls,
            **block_kwargs,
        )
    return nodes


class CyGFlow(nn.Module):
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
    ):
        super(CyGFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in [const.BACKBONE_RESNET18, const.BACKBONE_VSSM_SMALL]:
            # Replace the original ResNet18 visual encoder with VMamba-VSSM.
            # Multi-scale features -> normalizing flows.
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

            # Select VSSM stages from config so we can run strict ablations
            # under one identical code path (e.g. [0,1,2], [1,2,3], [0,1,2,3]).
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
            # for resnets / other CNN-like feature extractors, create trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )
        elif backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
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
                nf_cyg_flow(
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
                )
            )
        self.use_saf = use_saf
        if self.use_saf:
            self.saf_blocks = nn.ModuleList([SAF(c) for c in channels])
        else:
            self.saf_blocks = None

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
                        "fusion_init length {} must match number of scales {}".format(
                            len(fusion_init), n
                        )
                    )
                t = torch.tensor([float(w) for w in fusion_init], dtype=torch.float32)
                t = t / t.sum()
                logits = torch.log(t + 1e-8)
            else:
                logits = torch.zeros(n, dtype=torch.float32)
            self.fusion_logits = nn.Parameter(logits)
        else:
            if fusion_weights is None:
                weights = [1.0 / len(channels)] * len(channels)
            else:
                if len(fusion_weights) != len(channels):
                    raise ValueError(
                        "fusion_weights length {} must match number of scales {}".format(
                            len(fusion_weights), len(channels)
                        )
                    )
                weights = [float(w) for w in fusion_weights]
                s = sum(weights)
                if s <= 0:
                    raise ValueError("fusion_weights sum must be > 0")
                weights = [w / s for w in weights]
            self.register_buffer(
                "fusion_weights", torch.tensor(weights, dtype=torch.float32)
            )
        self.input_size = input_size

    def _apply_fusion_norm_power(self, x):
        if self.fusion_pre_norm_power == 1.0:
            return x
        return torch.pow(torch.clamp(x, min=0.0), self.fusion_pre_norm_power)

    def _apply_fusion_pre_norm(self, anomaly_map_list):
        # Normalize each scale independently before weighted fusion to reduce
        # dynamic-range imbalance across scales (A2 ablation).
        if self.fusion_pre_norm == "none":
            return anomaly_map_list

        if self.fusion_pre_norm == "minmax":
            x_min = anomaly_map_list.amin(dim=(2, 3), keepdim=True)
            x_max = anomaly_map_list.amax(dim=(2, 3), keepdim=True)
            x = (anomaly_map_list - x_min) / (x_max - x_min + self.fusion_pre_norm_eps)
            x = torch.clamp(x, 0.0, 1.0)
            return self._apply_fusion_norm_power(x)

        if self.fusion_pre_norm == "robust":
            b, _, h, w, s = anomaly_map_list.shape
            flat = anomaly_map_list.permute(0, 4, 1, 2, 3).reshape(b, s, -1)
            q_low = torch.quantile(flat, self.fusion_pre_norm_low_q, dim=2, keepdim=True)
            q_high = torch.quantile(flat, self.fusion_pre_norm_high_q, dim=2, keepdim=True)
            flat = (flat - q_low) / (q_high - q_low + self.fusion_pre_norm_eps)
            flat = self._apply_fusion_norm_power(torch.clamp(flat, 0.0, 1.0))
            return flat.view(b, s, 1, h, w).permute(0, 2, 3, 4, 1)

        # zscore
        mean = anomaly_map_list.mean(dim=(2, 3), keepdim=True)
        std = anomaly_map_list.std(dim=(2, 3), keepdim=True, unbiased=False)
        z = (anomaly_map_list - mean) / (std + self.fusion_pre_norm_eps)
        if self.fusion_pre_norm_clip > 0:
            z = torch.clamp(z, -self.fusion_pre_norm_clip, self.fusion_pre_norm_clip)
        return z

    def forward(self, x):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        if self.use_saf:
            features = [self.saf_blocks[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            prob_map_list = []
            anomaly_map_list = []
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                prob_up = F.interpolate(
                    prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                prob_map_list.append(prob_up)
                anomaly_map_list.append(-prob_up)

            if self.fusion_type == "min":
                # Conservative normality estimate: pixel-wise minimum normal probability.
                prob_stack = torch.stack(prob_map_list, dim=-1)  # (B,1,H,W,S)
                fused_prob_normal = torch.min(prob_stack, dim=-1).values
                anomaly_map = -fused_prob_normal
            elif self.fusion_type == "cross_attn":
                # Lightweight cross-scale attention at each pixel.
                anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)  # (B,1,H,W,S)
                anomaly_map_list = self._apply_fusion_pre_norm(anomaly_map_list)
                tokens = anomaly_map_list.squeeze(1)  # (B,H,W,S)
                s = tokens.shape[-1]
                attn = torch.softmax(tokens / (float(s) ** 0.5), dim=-1)  # (B,H,W,S)
                context = attn * tokens  # (B,H,W,S)

                if self.fusion_learnable:
                    w = F.softmax(self.fusion_logits, dim=0).to(context.device)
                else:
                    w = self.fusion_weights.to(context.device)
                fused = torch.sum(context * w.view(1, 1, 1, -1), dim=-1, keepdim=True)
                anomaly_map = fused.permute(0, 3, 1, 2).contiguous()
            else:
                anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)  # (B,1,H,W,S)
                anomaly_map_list = self._apply_fusion_pre_norm(anomaly_map_list)
                if self.fusion_learnable:
                    w = F.softmax(self.fusion_logits, dim=0).to(anomaly_map_list.device)
                    weights = w.view(1, 1, 1, 1, -1)
                else:
                    weights = self.fusion_weights.to(anomaly_map_list.device).view(
                        1, 1, 1, 1, -1
                    )
                anomaly_map = torch.sum(anomaly_map_list * weights, dim=-1)
            ret["anomaly_map"] = anomaly_map
        return ret
