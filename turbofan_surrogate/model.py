"""Flax MLP surrogate for the ODSMR turbofan thermodynamic map.

Approximates
    f : (state in R^10, ALT, MACH, COMMAND, DTAMB, phase in {MTO,MCL,CR})
        -> (HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin,
            Fuel_flow, HPC_Pout_st, LP_Nmech)

The Surrogate wrapper carries per dim input/output normalisation stats
as float32 numpy arrays, applies them at inference time, and exposes a
batched `.predict(x_raw)` that returns sensors in natural units. Saved
artefacts are single .pkl files containing weights + stats + config.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from turbofan_surrogate.constants import OUTPUT_DIM


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Plain MLP operating in normalised input/output space."""
    hidden: int = 384
    depth: int = 4
    output_dim: int = OUTPUT_DIM

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden)(x)
            x = nn.silu(x)
        return nn.Dense(self.output_dim)(x)


# ---------------------------------------------------------------------------
# Surrogate wrapper (model + normalisation stats + predict helper)
# ---------------------------------------------------------------------------
@dataclass
class Surrogate:
    """Trained surrogate ready for inference.

    `predict(x_raw)` accepts raw 17D inputs (see `inference.build_inputs`)
    and returns 7D sensor predictions in natural units.
    """
    model: MLP
    params: object
    input_mean: np.ndarray
    input_std:  np.ndarray
    output_mean: np.ndarray
    output_std:  np.ndarray

    def __post_init__(self):
        @jax.jit
        def _forward(params, x_norm):
            return self.model.apply(params, x_norm)
        self._forward = _forward

    def predict(self, x_raw: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Run prediction in batches; returns (n, 7) float32 in natural units."""
        x_raw = np.asarray(x_raw, dtype=np.float32)
        x_norm_full = (x_raw - self.input_mean) / self.input_std

        out = np.empty((len(x_raw), OUTPUT_DIM), dtype=np.float32)
        for i in range(0, len(x_raw), batch_size):
            x_chunk = jnp.asarray(x_norm_full[i:i + batch_size])
            y_chunk = self._forward(self.params, x_chunk)
            y_chunk.block_until_ready()
            out[i:i + batch_size] = np.asarray(y_chunk)
        return out * self.output_std + self.output_mean


# ---------------------------------------------------------------------------
# (de)serialisation
# ---------------------------------------------------------------------------
def save_surrogate(path: str | Path, surrogate: Surrogate) -> None:
    """Persist weights + normalisation stats + model config to a pickle."""
    payload = {
        "params":      jax.tree_util.tree_map(lambda v: np.asarray(v),
                                              surrogate.params),
        "input_mean":  np.asarray(surrogate.input_mean),
        "input_std":   np.asarray(surrogate.input_std),
        "output_mean": np.asarray(surrogate.output_mean),
        "output_std":  np.asarray(surrogate.output_std),
        "config": {
            "hidden":     surrogate.model.hidden,
            "depth":      surrogate.model.depth,
            "output_dim": surrogate.model.output_dim,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_surrogate(path: str | Path) -> Surrogate:
    """Load a surrogate previously saved with `save_surrogate`."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    cfg   = payload["config"]
    model = MLP(hidden=cfg["hidden"], depth=cfg["depth"],
                output_dim=cfg["output_dim"])
    return Surrogate(
        model       = model,
        params      = payload["params"],
        input_mean  = payload["input_mean"].astype(np.float32),
        input_std   = payload["input_std"].astype(np.float32),
        output_mean = payload["output_mean"].astype(np.float32),
        output_std  = payload["output_std"].astype(np.float32),
    )
