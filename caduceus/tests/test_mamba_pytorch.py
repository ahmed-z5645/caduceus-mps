"""Smoke tests for the pure-PyTorch drop-in in caduceus/mamba_pytorch.py.

Confirms: shape contract, state-dict key parity with mamba_ssm upstream,
RMSNorm correctness, and Block residual semantics.
"""
import pytest
import torch
from torch import nn

from caduceus.mamba_pytorch import Block, Mamba, RMSNorm, layer_norm_fn, rms_norm_fn


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_fused_norm_fns_are_none():
    """Existing try/except in modeling_caduceus.py expects these may be None."""
    assert layer_norm_fn is None
    assert rms_norm_fn is None


def test_mamba_shape():
    device = _device()
    torch.manual_seed(0)
    m = Mamba(d_model=64, device=device)
    x = torch.randn(2, 128, 64, device=device)
    y = m(x)
    assert y.shape == (2, 128, 64)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_mamba_inference_params_rejected():
    m = Mamba(d_model=32)
    x = torch.randn(1, 16, 32)
    with pytest.raises(NotImplementedError):
        m(x, inference_params=object())


def test_mamba_state_dict_keys_match_upstream():
    """The exact key set produced by mamba_ssm.modules.mamba_simple.Mamba(d_model=256,
    d_state=16, d_conv=4, expand=2, bias=False, conv_bias=True). Verified against
    the safetensors header of kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
    in docs/MPS_PORT_NOTES.md.
    """
    m = Mamba(d_model=256, d_state=16, d_conv=4, expand=2, bias=False, conv_bias=True)
    expected = {
        "in_proj.weight",
        "conv1d.weight",
        "conv1d.bias",
        "x_proj.weight",
        "dt_proj.weight",
        "dt_proj.bias",
        "A_log",
        "D",
        "out_proj.weight",
    }
    assert set(m.state_dict().keys()) == expected

    # Shape spot-checks (d_inner=512, dt_rank=ceil(256/16)=16)
    sd = m.state_dict()
    assert sd["in_proj.weight"].shape == (1024, 256)
    assert sd["out_proj.weight"].shape == (256, 512)
    assert sd["conv1d.weight"].shape == (512, 1, 4)
    assert sd["x_proj.weight"].shape == (48, 512)
    assert sd["dt_proj.weight"].shape == (512, 16)
    assert sd["A_log"].shape == (512, 16)
    assert sd["D"].shape == (512,)


def test_mamba_state_dict_keys_with_bias_true():
    """When bias=True, in_proj.bias and out_proj.bias appear."""
    m = Mamba(d_model=64, bias=True)
    keys = set(m.state_dict().keys())
    assert "in_proj.bias" in keys
    assert "out_proj.bias" in keys


def test_rmsnorm_no_bias_param():
    """RMSNorm must register bias=None so state_dict has no 'bias' key.
    The pretrained Caduceus checkpoint does not contain norm.bias entries.
    """
    n = RMSNorm(64)
    assert n.bias is None
    assert "bias" not in n.state_dict()
    assert set(n.state_dict().keys()) == {"weight"}


def test_rmsnorm_matches_reference():
    """Compare against an explicit reference computation."""
    torch.manual_seed(0)
    d, eps = 32, 1e-5
    n = RMSNorm(d, eps=eps)
    n.weight.data.copy_(torch.randn(d))
    x = torch.randn(4, 16, d)
    out = n(x)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * n.weight
    assert torch.allclose(out, ref, atol=1e-6)


def test_rmsnorm_matches_nn_rmsnorm():
    """Sanity: matches torch.nn.RMSNorm when available (PyTorch >= 2.4)."""
    if not hasattr(nn, "RMSNorm"):
        pytest.skip("torch.nn.RMSNorm requires PyTorch >= 2.4")
    torch.manual_seed(0)
    d = 64
    ours = RMSNorm(d, eps=1e-5)
    ref = nn.RMSNorm(d, eps=1e-5)
    ours.weight.data.copy_(ref.weight.data)
    x = torch.randn(2, 8, d)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)


def test_block_first_layer_residual_is_input():
    """When residual is None on entry, the returned residual equals the input."""
    torch.manual_seed(0)
    dim = 32
    mixer_cls = lambda d: Mamba(d_model=d)  # noqa: E731
    blk = Block(dim, mixer_cls, norm_cls=RMSNorm, fused_add_norm=False)
    x = torch.randn(1, 16, dim)
    _, residual = blk(x, residual=None)
    assert torch.equal(residual, x)


def test_block_subsequent_layer_residual_accumulates():
    """When residual is provided, the returned residual is hidden_states + residual."""
    torch.manual_seed(0)
    dim = 32
    blk = Block(dim, lambda d: Mamba(d_model=d), norm_cls=RMSNorm)
    x = torch.randn(1, 8, dim)
    r = torch.randn(1, 8, dim)
    _, residual_out = blk(x, residual=r)
    assert torch.allclose(residual_out, x + r)


def test_block_rejects_fused_add_norm():
    with pytest.raises(RuntimeError, match="fused_add_norm"):
        Block(16, lambda d: Mamba(d_model=d), fused_add_norm=True)
