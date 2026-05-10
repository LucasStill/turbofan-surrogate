# turbofan-surrogate

Neural surrogate for the ODSMR turbofan engine simulator. A small Flax MLP
(20k to 1M parameters) approximates ODSMR's thermodynamic map and runs
on the order of a million times faster per device than the original
CPU simulator, with overall R² above 0.998 on every sensor.

The surrogate is intended as a fast inference backend for reinforcement
learning, world model evaluation, and any pipeline that needs to call
ODSMR many millions of times.

## What it does

Takes a 17 dimensional input

  - 10 dim health state (per module efficiency and mass flow degradation)
  - altitude, Mach, thrust command, ambient temperature delta
  - 3 dim phase one hot (MTO take off, MCL climb, CR cruise)

and returns 7 sensor readings

  - `HPC_Tout` (K), `HP_Nmech` (rpm), `HPC_Tin` (K), `LPT_Tin` (K),
    `Fuel_flow` (kg/s), `HPC_Pout_st` (Pa), `LP_Nmech` (rpm)

## Quick start

```bash
pip install -e .
# for GPU inference (Linux + CUDA 12 + Ampere/Hopper/Volta):
pip install -e ".[gpu]"
```

```python
import numpy as np
from turbofan_surrogate import (
    load_surrogate, build_inputs, CONTEXT_PHASES, SENSOR_NAMES,
)

# Load the recommended model
surrogate = load_surrogate("weights/default.pkl")

# Build a batch of inputs
n = 1024
states = np.random.uniform(-0.05, 0.0, (n, 10)).astype("float32")
ci     = np.random.randint(0, 16, n)            # 16 named contexts
dtamb  = np.full(n, 10.0, dtype="float32")
alt    = np.full(n, 35000.0, dtype="float32")
mach   = np.full(n, 0.78, dtype="float32")
cmd    = np.full(n, 25000.0, dtype="float32")
X      = build_inputs(states, ci, dtamb, alt, mach, cmd)

# Predict (7 sensors per input row, in natural units)
sensors = surrogate.predict(X)
print(dict(zip(SENSOR_NAMES, sensors[0])))
```

A complete worked example with timing is in `examples/inference_demo.py`.

## Available variants

Four architectures are shipped under `weights/`. All four were trained on
the same 999k row dataset for 100 epochs.

| Variant | Hidden × Depth | Params | Disk | nMAE (mean) | Throughput @ batch 16k (V100) |
|---|---|---|---|---|---|
| `tiny` | 128 × 2 | 19.7k | 78 KB | 0.0137 | 6.1M samples/sec |
| `small` | 256 × 3 | 138k | 540 KB | 0.0109 | 6.1M samples/sec |
| **`default`** | 384 × 4 | 453k | 1.8 MB | **0.0055** | 3.8M samples/sec |
| `large` | 512 × 5 | 1.06M | 4.1 MB | 0.0039 | 3.0M samples/sec |

Recommendations:

- For most uses, ship **`default`**. R² ≥ 0.9998 on every sensor, mean
  nMAE 0.55 % of natural sensor std, 3.8M samples/sec on V100.
- For maximum throughput, ship **`small`**. Identical throughput to
  `tiny` but 17 % lower nMAE, only 540 KB on disk.
- For best accuracy, ship **`large`**. About 30 % lower nMAE than
  default at 21 % throughput cost.
- `tiny` is dominated by `small` and is included for completeness only.

A full per sensor breakdown lives in `reports/comparison.md` and a full
held out evaluation report (per phase, per context, per wear mode, per
quantile bin, hardest cases, sensitivity matrix, throughput) for the
default model lives in `reports/default_evaluation.md`.

## Reference numbers

On the held out 200k row eval set, the default model achieves:

| Sensor | nMAE (×σ) | R² | Max abs error |
|---|---|---|---|
| HPC_Tout | 0.0053 | 0.9999 | 9.6 K |
| HP_Nmech | 0.0102 | 0.9998 | 263 rpm |
| HPC_Tin | 0.0044 | 1.0000 | 8.2 K |
| LPT_Tin | 0.0056 | 0.9999 | 22 K |
| Fuel_flow | 0.0028 | 1.0000 | 0.024 kg/s |
| HPC_Pout_st | 0.0028 | 1.0000 | 72 kPa |
| LP_Nmech | 0.0073 | 0.9999 | 129 rpm |

Compared against the per context mean baseline (a strong baseline that
already knows the operating point, just not the state), the trained
surrogate is 14× to 64× lower nMAE per sensor — i.e. the model learned
state-driven physics, not just the per context mean.

## Operating envelope

The surrogate was trained on the TurboSens operating distribution, which
is wider than ODSMR's documented "tested core". Inputs should fall
within:

- ALT: 0 to 39000 ft
- MACH: 0 to 0.84
- COMMAND: 15000 to 125000 lbf (used only for CR; ignored in MTO/MCL)
- DTAMB: -15 to +30 K
- state: each entry within `STATE_BOUNDS` (Eff in [-0.05, 0],
  Wc in [-0.05, 0.03] or [-0.05, 0.05] depending on module)

Predictions outside this envelope are not characterised; outputs may
extrapolate poorly. Inside the envelope, see `MODEL_CARD.md` for slice
level error guarantees.

## Training data and methodology

The training set is 999k samples drawn from a four mode wear mixture
(pristine / mid wear / heavy wear / post action restoration) crossed
with uniform context sampling and per slot ALT/MACH/COMMAND jitter,
then run through ODSMR. The dataset is not redistributed here; see the
companion `rl_simulator_safran` repository for the generation pipeline.

Training: 100 epochs, AdamW (lr 1e-3 → 1e-5 cosine, weight decay 1e-4),
per output normalised MSE in float32, batch 8192. Wall time on a single
V100 is 15 to 75 seconds depending on variant.

## Model card

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use, scope,
known failure modes, and ethical considerations.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If this surrogate is useful in your work, please cite the companion
TurboSens paper (NeurIPS 2026 submission, citation TBD).
