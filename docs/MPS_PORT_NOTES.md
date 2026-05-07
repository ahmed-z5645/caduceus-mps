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

`Mamba` must produce a state-dict whose keys exactly match `mamba_ssm.modules.mamba_simple.Mamba` so HuggingFace pretrained weights load without re-keying.

# Step 2 findings: state-dict and constructor compatibility

Sources read:
- `mamba_ssm.modules.mamba_simple.Mamba` at `state-spaces/mamba` tag `v1.2.0.post1` (the version pinned in `caduceus_env.yml:50`).
- `mamba_ssm.ops.triton.layernorm.RMSNorm` at the same tag.
- `johnma2006/mamba-minimal` `model.py` (master branch).
- The safetensors header from the pretrained checkpoint `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16` (274 keys, fp32).

## State-dict key contract (verified against the actual pretrained checkpoint)

The Caduceus-Ph checkpoint stores 274 fp32 tensors. Per-layer pattern for `d_model=256, d_inner=512, d_state=16, dt_rank=16` (auto = ceil(256/16)):

```
caduceus.backbone.embeddings.word_embeddings.weight  [16, 256]
caduceus.backbone.norm_f.weight                       [256]
caduceus.backbone.layers.{i}.norm.weight              [256]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.A_log              [512, 16]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.D                  [512]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.conv1d.weight      [512, 1, 4]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.conv1d.bias        [512]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.dt_proj.weight     [512, 16]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.dt_proj.bias       [512]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.in_proj.weight     [1024, 256]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.out_proj.weight    [256, 512]
caduceus.backbone.layers.{i}.mixer.mamba_fwd.x_proj.weight      [48, 512]
caduceus.backbone.layers.{i}.mixer.mamba_rev.A_log              [512, 16]
caduceus.backbone.layers.{i}.mixer.mamba_rev.D                  [512]
caduceus.backbone.layers.{i}.mixer.mamba_rev.conv1d.weight      [512, 1, 4]
caduceus.backbone.layers.{i}.mixer.mamba_rev.conv1d.bias        [512]
caduceus.backbone.layers.{i}.mixer.mamba_rev.dt_proj.weight     [512, 16]
caduceus.backbone.layers.{i}.mixer.mamba_rev.dt_proj.bias       [512]
caduceus.backbone.layers.{i}.mixer.mamba_rev.x_proj.weight      [48, 512]
```

Notes:
- **No `in_proj.bias` / `out_proj.bias`**: the pretrained config sets `ssm_cfg.bias=false`, so `in_proj` and `out_proj` are biasless `nn.Linear`s.
- **`mamba_rev` has 7 keys vs `mamba_fwd`'s 9**: `bidirectional_weight_tie=true` shares `in_proj.weight` and `out_proj.weight` between the two. `BiMambaWrapper.__init__` (`caduceus/modeling_caduceus.py:115-118`) re-ties them after instantiation, so PyTorch's state dict serializes the shared parameters under `mamba_fwd.*` only.
- **No `norm.bias`**: see RMSNorm note below.
- `dt_proj.weight` is `[d_inner, dt_rank]` because the upstream `Linear` is `nn.Linear(dt_rank, d_inner)` (PyTorch stores Linear weights as `[out, in]`).

Total per layer: 17 tensor keys (9 + 7 + 1). 16 layers × 17 = 272 + 2 top-level = 274. ✓

## mamba-minimal `MambaBlock` parameter names (verified line-by-line)

| Param | mamba-minimal `MambaBlock` (`model.py:177-203`) | Upstream `Mamba` (`mamba_simple.py:62-117`) | Match? |
|---|---|---|---|
| `in_proj.weight`, optional `in_proj.bias` | `nn.Linear(d_model, d_inner*2, bias=args.bias)` (l. 183) | `nn.Linear(d_model, d_inner*2, bias=bias, **factory_kwargs)` (l. 62) | ✓ |
| `conv1d.weight`, `conv1d.bias` | `nn.Conv1d(d_inner, d_inner, bias=conv_bias, kernel_size=d_conv, groups=d_inner, padding=d_conv-1)` (l. 185-192) | identical (l. 64-72) | ✓ |
| `x_proj.weight` | `nn.Linear(d_inner, dt_rank + d_state*2, bias=False)` (l. 195) | identical (l. 77-79) | ✓ |
| `dt_proj.weight`, `dt_proj.bias` | `nn.Linear(dt_rank, d_inner, bias=True)` (l. 198) | identical (l. 80) | ✓ |
| `A_log` | `nn.Parameter(torch.log(repeat(arange(1, d_state+1), 'n -> d n', d=d_inner)))` (l. 200-201) | identical (l. 104-110) | ✓ |
| `D` | `nn.Parameter(torch.ones(d_inner))` (l. 202) | identical (l. 114) | ✓ |
| `out_proj.weight`, optional `out_proj.bias` | `nn.Linear(d_inner, d_model, bias=args.bias)` (l. 203) | identical (l. 117) | ✓ |

**Verdict: state-dict drops in directly. No remap needed.**

## RMSNorm contract (CORRECTION to Step 3 plan)

`mamba_ssm.ops.triton.layernorm.RMSNorm` (lines 481-499 of `layernorm.py`):

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)         # <-- NO bias parameter
        self.reset_parameters()                        # ones-init for weight
```

**The pretrained checkpoint has `norm.weight` only — no `norm.bias`.** The original Step 3 plan declared `self.bias = nn.Parameter(torch.zeros(d_model))`, which would add a phantom `bias` key to the state dict and break loading (`load_state_dict(strict=True)` would fail with "unexpected key").

**Corrected RMSNorm for `caduceus/mamba_pytorch.py`:**

```python
class RMSNorm(nn.Module):
    """Matches mamba_ssm.ops.triton.layernorm.RMSNorm: weight only, bias=None."""
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)  # exposed as None for fused_add_norm compat

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
```

This still satisfies `RCPSAddNormWrapper` (`caduceus/modeling_rcps.py:117`), which only reads `self.submodule.weight.dtype`. The `isinstance(self.norm, RMSNorm)` check (`modeling_rcps.py:155`) works because we export this class as `RMSNorm`. The fused path (`modeling_rcps.py:178-184`) reads `self.norm.bias` and would pass `None` to `rms_norm_fn`, but we never enter that path (we force `fused_add_norm=False` and `rms_norm_fn = None`).

## Constructor adapter

Upstream's `Mamba.__init__` signature (must be matched verbatim by our drop-in for `BiMambaWrapper` to work without changes):

```python
Mamba(d_model, d_state=16, d_conv=4, expand=2,
      dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random",
      dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False,
      use_fast_path=True, layer_idx=None, device=None, dtype=None)
```

mamba-minimal's `MambaBlock(args: ModelArgs)` consumes only a subset: `d_model, d_state, d_conv, expand, dt_rank, conv_bias, bias`. The remaining seven (`dt_min, dt_max, dt_init, dt_scale, dt_init_floor, use_fast_path, layer_idx`) are init-time hyperparameters that don't affect the forward pass once weights are loaded — our `Mamba.__init__` accepts them for signature compatibility and silently ignores them.

`device` and `dtype` are threaded through to `nn.Linear` / `nn.Conv1d` / `nn.Parameter` constructors via `factory_kwargs = {"device": device, "dtype": dtype}` — same pattern as upstream.

## Forward signature

Upstream: `forward(hidden_states, inference_params=None) -> Tensor`, shape `(B, L, D) -> (B, L, D)`.
mamba-minimal: `forward(x) -> Tensor`, same shape contract.

Our drop-in adds the `inference_params=None` arg and raises `NotImplementedError` if non-None (matches the inference-only scope; Caduceus's masked-LM path always passes `None`).

## Forward semantics — equivalence at fp32

The two forward implementations are mathematically identical at fp32, modulo numerical precision:

- mamba-minimal does an explicit Python loop over the L sequence positions in `selective_scan` (l. 317-321) — sequential and slow but correct.
- Upstream calls `selective_scan_fn` (CUDA Triton kernel) or `mamba_inner_fn` (when `use_fast_path=True`). Both compute the same recurrence in fp32 with parallel scan.

For inference at `seq_len ≤ 1024` on MPS the sequential loop is acceptable (~1s per forward at L=1024, d_model=256, n_layer=16). At L=131k it would take minutes — explicitly out of scope.

## What this means for Step 3

1. Vendor mamba-minimal's `MambaBlock` body into `caduceus/mamba_pytorch.Mamba`, but with the upstream constructor signature and the fixed RMSNorm above.
2. No state-dict remap required — the checkpoint loads as-is.
3. `Block` only needs to replicate the residual+norm+mixer pattern (rcps=False checkpoints use it; Caduceus-Ph is rcps=False). The fused_add_norm path is unreachable on this build.
4. `layer_norm_fn = None`, `rms_norm_fn = None` in the module, exploiting the existing try/except fallback in `modeling_caduceus.py:21-27` and `modeling_rcps.py:12-18`.
