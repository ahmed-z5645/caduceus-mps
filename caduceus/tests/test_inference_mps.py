"""End-to-end smoke test: load pretrained Caduceus from HF Hub and run inference on MPS.

Requires network access to download the checkpoint on first run.
"""
import os

import pytest
import torch

from caduceus.modeling_caduceus import CaduceusForMaskedLM
from caduceus.tokenization_caduceus import CaduceusTokenizer

MODEL_ID = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
SEQ = "ACGT" * 32  # 128 nt

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable"
)
requires_network = pytest.mark.skipif(
    os.environ.get("NO_NETWORK_TESTS") == "1", reason="NO_NETWORK_TESTS=1 set"
)


@pytest.fixture(scope="module")
def loaded_model():
    """Load the Caduceus-Ph checkpoint with fused_add_norm=False overridden."""
    model = CaduceusForMaskedLM.from_pretrained(MODEL_ID, fused_add_norm=False).eval()
    return model


@pytest.fixture(scope="module")
def tokenized_seq():
    tok = CaduceusTokenizer(model_max_length=131072)
    return torch.tensor([tok(SEQ)["input_ids"]], dtype=torch.long)


@requires_network
def test_pretrained_weights_tied(loaded_model):
    """bidirectional_weight_tie=True must survive from_pretrained's meta init."""
    layer0 = loaded_model.caduceus.backbone.layers[0].mixer
    fwd_ptr = layer0.mamba_fwd.in_proj.weight.data_ptr()
    rev_ptr = layer0.mamba_rev.in_proj.weight.data_ptr()
    assert fwd_ptr == rev_ptr, "mamba_rev.in_proj should be tied to mamba_fwd.in_proj"
    assert torch.isfinite(layer0.mamba_rev.in_proj.weight).all()


@requires_network
def test_pretrained_inference_cpu(loaded_model, tokenized_seq):
    """Forward pass on CPU with the loaded weights produces finite logits of the expected shape."""
    with torch.no_grad():
        out = loaded_model(tokenized_seq)
    vocab = loaded_model.config.vocab_size
    assert out.logits.shape == (1, tokenized_seq.shape[1], vocab)
    assert torch.isfinite(out.logits).all()


@requires_network
@requires_mps
def test_pretrained_inference_mps(loaded_model, tokenized_seq):
    """Forward pass on MPS produces finite logits matching CPU within fp32 tolerance."""
    with torch.no_grad():
        out_cpu = loaded_model(tokenized_seq)
    model_mps = loaded_model.to("mps")
    try:
        with torch.no_grad():
            out_mps = model_mps(tokenized_seq.to("mps"))
        assert torch.isfinite(out_mps.logits).all()
        max_diff = (out_cpu.logits - out_mps.logits.cpu()).abs().max().item()
        # 1e-3 is loose enough to absorb MPS fp32 dispatch noise across 16 layers, tight
        # enough to catch a real algorithmic divergence.
        assert max_diff < 1e-3, f"CPU vs MPS max abs diff {max_diff} exceeds tolerance"
    finally:
        loaded_model.to("cpu")  # restore for other tests


@requires_network
@requires_mps
def test_pretrained_top1_recovers_input(loaded_model, tokenized_seq):
    """The masked-LM should predict each unmasked nucleotide as itself with high accuracy.

    For a competent DNA LM, the top-1 token at an unmasked position should usually be the
    input token (the model has learned base identity). We require >=90% on the simple
    ACGT-repeat sequence; in practice it's 100% on 128/128 non-CLS positions.
    """
    model_mps = loaded_model.to("mps")
    try:
        with torch.no_grad():
            out = model_mps(tokenized_seq.to("mps"))
        top1 = out.logits[0].argmax(dim=-1).cpu()
        # Skip position 0 (CLS) — the model has no reason to predict it as a nucleotide.
        matches = (top1[1:] == tokenized_seq[0, 1:]).sum().item()
        n = tokenized_seq.shape[1] - 1
        assert matches / n >= 0.90, f"top-1 recovery only {matches}/{n}"
    finally:
        loaded_model.to("cpu")
