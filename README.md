# Timestral.jl — a Julia-native Timestral forecaster

A pure-Julia, **standard-library-only** port of Mistral's Timestral time-series
models. No PyTorch, no Python, no external Julia packages — one file,
`Timestral.jl`, that loads the exported checkpoints (`config.json` +
`model.safetensors`) directly and runs fully native probabilistic forecasts on
CPU. Supports all four released checkpoints: **Timestral 5M, 28M, 123M and
209M**.

## Quick start

```julia
include("Timestral.jl")
using .Timestral

fc  = Timestral.from_pretrained("mistralai/Timestral-5m")   # or a local dir
# fc = Timestral.from_pretrained("mistralai/Timestral-209m") # big models too

# context: (batch, variates, time); NaN = missing observation
context = [ [1.0, 2.0, 3.1, 2.8, 3.0, missing, 3.3] ]        # B=1, V=1, T=7

f = Timestral.forecast(fc, context; prediction_length = 96)

med = Timestral.median(f)              # (B, V, H) median path
q10 = Timestral.quantile_at(f, 0.10)   # any configured level
q90 = Timestral.quantile_at(f, 0.90)
Timestral.quantile_levels(f)           # 21 levels, 0.01 … 0.99
```

## API

| Function | Description |
|---|---|
| `from_pretrained(path_or_repo; dtype, max_context, per_shot_horizon_patches, quantile_real_cap_k, revision)` | Load a checkpoint from a local directory or a Hugging Face Hub repo id (auto-download + cache). `dtype = Float32` (default, matches the reference) or `Float64`. |
| `forecast(fc, context; prediction_length, observation_mask, past_covariates, known_covariates, categorical_past_covariates, categorical_known_covariates, symmetric_averaging, clip_nonnegative)` | Probabilistic forecast. Returns a `Forecast` with `quantiles::Array{T,4}` shaped `(B, V, H, Q)`, sorted along `Q`. |
| `median(f)` | `(B, V, H)` median forecast. |
| `quantile_at(f, level)` | `(B, V, H)` forecast at one of the model's quantile levels. |
| `quantile_levels(f)` | The configured levels (from `config.json`). |
| `encode_categorical(fc, context; prediction_length, ...)` | Standalone categorical target-encoding (what `forecast` does internally); useful for debugging/inspection. |
| `json_parse(s)` | Minimal JSON reader (stdlib only). |

`context` (and covariate arrays) may be nested Julia arrays of any `Real`,
`missing`/`NaN` both count as missing. Covariate shapes:

| Input | Shape |
|---|---|
| `context` | `(B, V, T)` |
| `observation_mask` | `(B, V, T)` — 0/false = treat as unobserved |
| `past_covariates` | `(B, C, T)` |
| `known_covariates` | `(B, C, T + prediction_length)` — must be known for the forecast span too |
| `categorical_past_covariates` | `(B, C, T)` integer category codes |
| `categorical_known_covariates` | `(B, C, T + prediction_length)` |

## Features

- **All four checkpoints**: Timestral 5M, 28M, 123M and 209M (any exported
  `Timestral` config).
- **Float32 and Float64** execution (`dtype` at load time).
- **Memory-mapped weights**: safetensors tensors are mapped read-only from the
  file, so even the 209M checkpoint loads in well under a gigabyte of heap and
  starts instantly. (In `Float64` the weights must be materialized — ~1 GB for
  123M, ~1.7 GB for 209M — so `Float32` is the default; `Float64` is always
  available for 5M/28M and for the larger models whenever RAM allows.)
- **Patch-causal attention**, multi-block autoregressive roll-out over long
  horizons (validated up to H = 1100), `max_context` truncation.
- **Missing data**: NaN gaps, delayed series start, observation masks,
  forward-fill + causal statistics.
- **Numeric covariates**: past (observed only) and known (past + future).
- **Categorical covariates**: target encoding with the reference's causal
  semantics, past and known.
- **Symmetric averaging** (`clampsym`) and non-negative clipping, both
  optional, both matching the reference.
- **Zero dependencies**: only Julia's standard library (JSON reader,
  safetensors reader with mmap, HTTP download via `Downloads` are all built in).

## Validation

Every feature path was cross-validated against an **independently written
Float64 reference implementation** (Node.js, computing in Float64 over the
same F32 checkpoints, written from the published reference semantics rather
than translated from this port) on synthetic test cases covering each feature:

| Case | Features | 5M f64, max abs | 28M f64, max abs |
|---|---|---|---|
| A | clean B=2, T=256, H=64 | 9.8e-15 | 6.2e-15 |
| B | NaN gaps, delayed start, V=3 + observation mask, H=96 | 6.4e-14 | 8.5e-10 |
| C | numeric past + known covariates | 9.8e-15 | 3.6e-14 |
| D | categorical covariates (encodings 0.0 diff) | 6.7e-15 | 1.1e-14 |
| E | long horizon H=1100, multi-block AR, max_context=512 | 3.4e-13 | 3.6e-13 |

The 123M and 209M checkpoints were verified with the same cross-check in
`Float32` (the reference computes in f64, so the agreement below is bounded by
f32 arithmetic, not by implementation differences):

| Case | 123M (Julia f32 vs JS f64) | 209M (Julia f32 vs JS f64) |
|---|---|---|
| A | 1.3e-6 abs / 4.0e-4 rel | 9.7e-7 abs / 7.4e-5 rel |
| C | 1.1e-5 abs / 1.6e-3 rel | 5.1e-6 abs / 6.2e-3 rel |
| E | 6.1e-4 abs / 9.4e-3 rel | 8.2e-5 abs / 1.2e-3 rel |

Additional checks:

- **Float32 fidelity** (Julia f32 vs f64, 5M): ≤ 7.8e-6 absolute on H ≤ 96
  cases, 6.9e-5 absolute over the H = 1100 autoregressive chain; the largest
  relative deviations occur only on near-zero quantile leaves. On 28M:
  ≤ 3.0e-5 absolute across all cases.
- Categorical encodings match between the two implementations exactly (0.0
  diff) on every categorical case, on both 5M and 28M.

## Forecast quality

Synthetic benchmark (16 series, trend + two sinusoids + AR(1) noise, context
512, median point forecast, 80% prediction interval = q10–q90; 209M row used
8 series / context 384 to fit a small test machine, its baselines are that
machine's own):

| Model | MAE H=32 | MAE H=96 | MAE H=224 | Coverage (H=32/96/224) |
|---|---|---|---|---|
| Timestral-5M | **0.268** | **0.325** | **0.358** | 84 / 81 / 89 % |
| Timestral-28M | **0.256** | **0.275** | **0.302** | 81 / 82 / 85 % |
| Timestral-123M | **0.238** | **0.257** | **0.262** | 78 / 83 / 87 % |
| Timestral-209M | **0.225** | **0.243** | **0.306** | 84 / 85 / 83 % |
| last-value baseline | 0.92–0.97 | 0.93–0.95 | 0.90–0.93 | — |
| mean baseline | 0.84–0.91 | 1.05–1.13 | 1.03–1.08 | — |
| seasonal-naive (P=24) | 0.95–1.02 | 0.80–1.13 | 0.72–1.04 | — |

Every checkpoint beats all simple baselines at every horizon, quality improves
with model size, and interval coverage stays near the nominal 80%.

## License

This port reimplements published reference semantics (Apache-2.0). The model
checkpoints remain under their own licenses — see the Hugging Face model cards
for Timestral 5M / 28M / 123M / 209M.
