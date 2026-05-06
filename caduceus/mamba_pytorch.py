"""Pure-PyTorch drop-in replacements for the four mamba_ssm symbols Caduceus uses.

Exports `Mamba`, `Block`, `RMSNorm`, `layer_norm_fn`, `rms_norm_fn` so that
`caduceus.modeling_caduceus` and `caduceus.modeling_rcps` can replace their
`mamba_ssm` imports with `from .mamba_pytorch import ...`.

Selective scan adapted from https://github.com/johnma2006/mamba-minimal (MIT).
The scan is a sequential Python loop over the sequence dimension - correct but
O(L) per step. Intended for inference at short lengths (<= ~4k tokens) on
CPU/MPS. State-dict keys match `mamba_ssm.modules.mamba_simple.Mamba` exactly,
so HuggingFace pretrained Caduceus weights load without re-keying.

See docs/MPS_PORT_NOTES.md for the full compatibility analysis.
"""
import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Matches mamba_ssm.ops.triton.layernorm.RMSNorm: weight only, bias=None."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class Mamba(nn.Module):
    """Drop-in for mamba_ssm.modules.mamba_simple.Mamba (forward only).

    Constructor accepts the full upstream kwarg list so BiMambaWrapper works
    unchanged. Init-time-only hyperparameters (dt_min/dt_max/dt_init/dt_scale/
    dt_init_floor/use_fast_path/layer_idx) are accepted and ignored - they
    affect only the random init, which is overwritten when pretrained weights
    load.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        A = torch.arange(1, self.d_state + 1, device=device, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: Tensor, inference_params=None) -> Tensor:
        if inference_params is not None:
            raise NotImplementedError(
                "MPS port supports inference_params=None only (no stateful decoding)."
            )
        b, l, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :l]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self._ssm(x)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x: Tensor) -> Tensor:
        d_inner, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, x)

        h = x.new_zeros(x.shape[0], d_inner, n)
        ys = []
        for i in range(x.shape[1]):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, i]))
        y = torch.stack(ys, dim=1)
        return y + x * D

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError("Stateful inference cache not supported on MPS port.")


class Block(nn.Module):
    """Drop-in for mamba_ssm.modules.mamba_simple.Block / mamba_ssm.modules.block.Block.

    Used by `create_block` when `rcps=False`. Pre-norm residual pattern matching
    upstream's non-fused branch:
        residual_out = hidden_states + (residual or 0)
        hidden_states = norm(residual_out)
        hidden_states = mixer(hidden_states)
    Returns (hidden_states, residual_out). The fused_add_norm branch is
    unreachable on this build (layer_norm_fn / rms_norm_fn are None).
    """

    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if fused_add_norm:
            raise RuntimeError(
                "fused_add_norm=True requires CUDA/Triton kernels which are unavailable "
                "on this build; set fused_add_norm=False."
            )

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


layer_norm_fn = None
rms_norm_fn = None
