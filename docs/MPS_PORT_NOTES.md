# MPS Port — Step 1 findings: the `mamba_ssm` surface in Caduceus

This document records exactly which `mamba_ssm` symbols Caduceus depends on, where they're imported, where they're called, and the constructor / forward signatures the replacement must match. It exists so the pure-PyTorch drop-in (`caduceus/mamba_pytorch.py`, written in Step 3) can be checked against a fixed target.

All file paths are relative to the repo root. Line numbers refer to the state of the repo at the start of the port (HEAD = `7840307` "Update README").

## Symbols imported from `mamba_ssm`

| Symbol | Imported at | Used at | Constructor args | Forward signature |
|---|---|---|---|---|
| `Mamba` | `caduceus/modeling_caduceus.py:11` | `BiMambaWrapper.__init__` (`caduceus/modeling_caduceus.py:105`, `:110`) | `d_model=…, **ssm_cfg` | `forward(hidden_states, inference_params=None) -> Tensor` of shape `(B, L, D)` |
| `Block` | `caduceus/modeling_caduceus.py:13` (v1) or `:15` (v2 fallback) | `create_block` when `rcps=False` (`caduceus/modeling_caduceus.py:64`) | `(d_model, mixer_cls, norm_cls=…, fused_add_norm=…, residual_in_fp32=…)`; v2 also takes `mlp_cls=nn.Identity` | `forward(hidden_states, residual=None, inference_params=None) -> (hidden_states, residual)` |
| `RMSNorm` | `caduceus/modeling_caduceus.py:22/25`, `caduceus/modeling_rcps.py:13/16`, `caduceus/tests/test_rcps.py:11/14` | `partial(... if not rms_norm else RMSNorm, eps=…)` (`caduceus/modeling_caduceus.py:62`, `:211`); `isinstance` check (`caduceus/modeling_rcps.py:155`) | `(d_model, eps=…, device=…, dtype=…)` | layer-norm-style `(B, L, D) -> (B, L, D)`; must expose `.weight`, `.bias`, `.eps` |
| `layer_norm_fn` / `rms_norm_fn` | same lines as `RMSNorm` | Only when `fused_add_norm=True` (`caduceus/modeling_caduceus.py:241–272`, `caduceus/modeling_rcps.py:175–195`) | — | `(input, weight, bias, residual=…, prenorm=…, residual_in_fp32=…, eps=…) -> Tensor` (or `(Tensor, Tensor)` if `prenorm=True`) |

## `ssm_cfg` knobs Caduceus actually sets

From `configs/model/caduceus.yaml` and the pretrained model config on HuggingFace Hub:

```
d_state=16
d_conv=4
expand=2
dt_rank="auto"
dt_min=0.001
dt_max=0.1
dt_init="random"
dt_scale=1.0
dt_init_floor=1e-4
conv_bias=True
bias=False
use_fast_path=True   # accepted by replacement, ignored
```

The replacement `Mamba.__init__` must accept all of these as keyword args (silently ignoring `use_fast_path`).

## What Caduceus does *not* use from `mamba_ssm`

- `Mamba2` — only `Mamba` (v1) is imported anywhere.
- `MambaConfig`, `MixerModel`, `_init_weights` from `mamba_ssm.models.*` — only used in `src/models/sequence/dna_embedding.py`, which is on the training path (out of scope for this port).
- `MambaLMHeadModel` — referenced as a string in `src/utils/registry.py:28`, never imported.
- `causal_conv1d_fn`, `selective_scan_fn`, `mamba_inner_fn` — not imported. Mamba's internal causal conv1d is reimplemented inside `Mamba.forward` and we replace the whole class.
- `triton` — not imported directly, only transitively via `mamba_ssm.ops.triton.*`.

## Crucial simplifications for the inference-only port

1. **`inference_params` is always `None`.** Every Caduceus forward path (masked-LM, embeddings) calls layers with `inference_params=None` (e.g. `caduceus/modeling_caduceus.py:230`). The replacement `Mamba` can require `inference_params is None` and `raise NotImplementedError` otherwise. `allocate_inference_cache` can be a stub that raises.
2. **`fused_add_norm=False` makes `layer_norm_fn` / `rms_norm_fn` unreachable.** The existing `try/except` in `caduceus/modeling_caduceus.py:21–27` already sets these to `None` if the import fails. The replacement module exports `layer_norm_fn = None` and `rms_norm_fn = None` and we set `fused_add_norm=False` everywhere it's parametrized.
3. **`Block` is only needed when `rcps=False`.** For the pretrained Caduceus-PS checkpoint (`rcps=True`) the wrapper used is `RCPSMambaBlock` from `caduceus/modeling_rcps.py:133`, which is already pure PyTorch. `Block` only matters for `rcps=False` checkpoints (e.g. Caduceus-Ph) and a minimal pre-norm residual implementation suffices.
4. **`RCPSEmbedding`, `RCPSWrapper`, `RCPSAddNormWrapper`, `RCPSMambaBlock`, `RCPSLMHead`, `BiMambaWrapper`, `CaduceusEmbeddings`, `CaduceusMixerModel`, `Caduceus`, `CaduceusForMaskedLM`** are already pure PyTorch and unchanged by this port — only the four imported symbols need replacing.

## Existing fallback behavior we exploit

`caduceus/modeling_caduceus.py:21–27`:
```python
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
```

The same shape exists in `caduceus/modeling_rcps.py:12–18` and `caduceus/tests/test_rcps.py:10–16`. We replace these import blocks with `from .mamba_pytorch import RMSNorm, layer_norm_fn, rms_norm_fn` (and additionally `Mamba, Block` in `modeling_caduceus.py`), preserving the contract that `layer_norm_fn` and `rms_norm_fn` may be `None`.

## Verification target for Step 3

`Mamba` must produce a state-dict whose keys exactly match `mamba_ssm.modules.mamba_simple.Mamba` so HuggingFace pretrained weights load without re-keying:

```
in_proj.weight, in_proj.bias
conv1d.weight, conv1d.bias
x_proj.weight
dt_proj.weight, dt_proj.bias
A_log              # shape (d_inner, d_state)
D                  # shape (d_inner,)
out_proj.weight, out_proj.bias
```

Step 2 confirms these match mamba-minimal's parameter names. If any key drifts, a remap is applied inside `caduceus/mamba_pytorch.Mamba._load_from_state_dict` rather than mutating the checkpoint.
