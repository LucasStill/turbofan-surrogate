"""SurrogateSimulator — drop-in replacement for an ODSMR call site.

Provides a `simulate(states, contexts) -> (N, 7)` method matching the
contract of any thermodynamic engine simulator that takes a list of
(state, context) pairs and returns sensor observations. Users replacing
ODSMR in an existing pipeline can swap their simulator object for an
instance of this class without changing any of the calling code.

Canonical import path:
    from turbofan_surrogate import SurrogateSimulator
    sim = SurrogateSimulator.from_hf("LucasThil/turbofan-surrogate")
    sensors = sim.simulate(states, contexts)   # (N, 7) float32

Pure-Python class. Depends only on `turbofan_surrogate.models.mlp`
(JAX / Flax) and on `huggingface_hub` for `from_hf` (lazy imported).

Note for the rl_simulator_safran tree
-------------------------------------
This is the same class that ships in the standalone release at
release/turbofan-surrogate/. The drop-in for the existing ZMQ-based
SimulatorClient lives at hpc.surrogate_simulator_client and now
re-exports this class for legacy callers; new code should import from
here (`from turbofan_surrogate import SurrogateSimulator`).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from turbofan_surrogate.models.mlp import (
    Surrogate,
    load_surrogate,
    phase_to_one_hot,
)


# ---------------------------------------------------------------------------
# Training-envelope reference (used for OOD warnings + sanity checks).
# These match scenarios.turbosens1.STATE_BOUNDS and turbosens2 envelopes.
# ---------------------------------------------------------------------------
_STATE_LO = np.full(10, -0.05, dtype=np.float32)
_STATE_HI = np.array(
    [0.0, 0.03,    # Bst  Eff, Wc
     0.0, 0.03,    # Fan
     0.0, 0.03,    # HPC
     0.0, 0.05,    # HPT
     0.0, 0.05],   # LPT
    dtype=np.float32,
)
_ALT_RANGE   = (-100.0,   40_000.0)   # feet (slight slack on bounds)
_MACH_RANGE  = (-0.01,    0.86)
_CMD_RANGE   = (1.0,      130_000.0)  # lbf (used only for CR)
_DTAMB_RANGE = (-20.0,    35.0)        # K

# Useful quick-reference constants for sanity checking. Healthy nominal
# cruise = state zero + the cruise mode's median context.
NOMINAL_HEALTHY_STATE = np.zeros(10, dtype=np.float32)
NOMINAL_CRUISE_CONTEXT: Dict = {
    "PHASE_TYPE": "CR",
    "DTAMB":      10.0,
    "ALT":        35_000.0,
    "MACH":       0.78,
    "COMMAND":    25_000.0,
}


def _build_inputs_from_contexts(
    states: np.ndarray, contexts: List[Dict],
) -> np.ndarray:
    """Map (states, list-of-context-dicts) onto the (N, 17) surrogate input."""
    n = len(states)
    if len(contexts) != n:
        raise ValueError(
            f"states and contexts length mismatch: "
            f"len(states)={n}, len(contexts)={len(contexts)}")

    phases = np.array([c["PHASE_TYPE"] for c in contexts])
    dtamb  = np.array([float(c["DTAMB"])    for c in contexts], dtype=np.float32)
    alt    = np.array([float(c["ALT"])      for c in contexts], dtype=np.float32)
    mach   = np.array([float(c["MACH"])     for c in contexts], dtype=np.float32)
    cmd    = np.array([float(c["COMMAND"])  for c in contexts], dtype=np.float32)

    state    = np.asarray(states, dtype=np.float32)
    cont     = np.stack([alt, mach, cmd, dtamb], axis=1).astype(np.float32)
    phase_oh = phase_to_one_hot(phases)
    return np.concatenate([state, cont, phase_oh], axis=1).astype(np.float32)


def _check_inputs(states: np.ndarray, contexts: List[Dict]) -> None:
    """Warn (once) if inputs fall meaningfully outside the training envelope.

    Loud about catching the most common usage trap: passing state values
    in [0, 1] instead of the [-0.05, 0.0] degradation deviations the
    surrogate was trained on. Predictions outside the envelope are
    arbitrary extrapolations and physically meaningless (sign flips,
    impossible magnitudes).
    """
    states = np.asarray(states)
    # Use 0.01 slack so legitimate edge samples don't trigger.
    if (states.min() < _STATE_LO.min() - 0.01) or \
       (states.max() > _STATE_HI.max() + 0.01):
        warnings.warn(
            "SurrogateSimulator: states fall outside the training envelope "
            f"(min={states.min():.4g}, max={states.max():.4g}; "
            f"trained range is approximately [-0.05, +0.05]). "
            "Predictions will extrapolate and may not be physically "
            "meaningful (sign flips, negative temperatures, etc). "
            "If you expected 'healthy nominal' use NOMINAL_HEALTHY_STATE "
            "(state=0) instead.",
            UserWarning, stacklevel=3,
        )

    valid_phases = ("MTO", "MCL", "CR")
    bad_phase = next((c["PHASE_TYPE"] for c in contexts
                      if c["PHASE_TYPE"] not in valid_phases), None)
    if bad_phase is not None:
        warnings.warn(
            f"SurrogateSimulator: PHASE_TYPE={bad_phase!r} not in "
            f"{valid_phases}. Phase one-hot will be all-zero, predictions "
            "are not characterised.",
            UserWarning, stacklevel=3,
        )

    alt_min  = min(float(c["ALT"])  for c in contexts)
    alt_max  = max(float(c["ALT"])  for c in contexts)
    mach_min = min(float(c["MACH"]) for c in contexts)
    mach_max = max(float(c["MACH"]) for c in contexts)
    if alt_min < _ALT_RANGE[0] or alt_max > _ALT_RANGE[1]:
        warnings.warn(
            f"SurrogateSimulator: ALT range [{alt_min:.0f}, {alt_max:.0f}] "
            f"is outside training envelope {list(_ALT_RANGE)} (feet). "
            "Predictions will extrapolate.",
            UserWarning, stacklevel=3,
        )
    if mach_min < _MACH_RANGE[0] or mach_max > _MACH_RANGE[1]:
        warnings.warn(
            f"SurrogateSimulator: MACH range [{mach_min:.3f}, {mach_max:.3f}] "
            f"is outside training envelope {list(_MACH_RANGE)}. "
            "Predictions will extrapolate.",
            UserWarning, stacklevel=3,
        )

    # Only check COMMAND for CR — for MTO/MCL the simulator ignores it.
    cr_cmds = [float(c["COMMAND"]) for c in contexts
               if c.get("PHASE_TYPE") == "CR"]
    if cr_cmds:
        cmin, cmax = min(cr_cmds), max(cr_cmds)
        if cmin < _CMD_RANGE[0] or cmax > _CMD_RANGE[1]:
            warnings.warn(
                f"SurrogateSimulator: COMMAND range [{cmin:.0f}, {cmax:.0f}] "
                f"on CR rows is outside training envelope {list(_CMD_RANGE)} "
                "(lbf). For nominal cruise use COMMAND=25000.",
                UserWarning, stacklevel=3,
            )


class SurrogateSimulator:
    """Stateful, callable surrogate exposing the ODSMR-style protocol.

    Parameters
    ----------
    weights_path:
        Local path to a `surrogate.pkl` produced by
        `turbofan_surrogate.training.train` (or downloaded from the
        published HF model repo).
    batch_size:
        Inference batch size for the underlying JAX forward.
    """

    def __init__(
        self, weights_path: str, batch_size: int = 8192,
        validate_inputs: bool = True,
    ) -> None:
        self._weights_path: str = str(Path(weights_path).resolve())
        self._batch_size:   int = batch_size
        self._validate:     bool = validate_inputs
        self._validated_once: bool = False
        self._surrogate:    Surrogate = load_surrogate(weights_path)
        self._closed:       bool = False

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        repo_id: str = "LucasThil/turbofan-surrogate",
        filename: str = "weights/default.pkl",
        *,
        batch_size: int = 8192,
        revision: Optional[str] = None,
    ) -> "SurrogateSimulator":
        """Pull weights from HuggingFace Hub, then instantiate.

        Requires `huggingface_hub` installed. The repo is public, so no
        token is needed. Cached under $HF_HOME (default `~/.cache/huggingface`).
        """
        from huggingface_hub import hf_hub_download   # lazy import
        path = hf_hub_download(
            repo_id   = repo_id,
            filename  = filename,
            repo_type = "model",
            revision  = revision,
        )
        return cls(weights_path=path, batch_size=batch_size)

    # ------------------------------------------------------------------
    # ODSMR-style protocol
    # ------------------------------------------------------------------
    def simulate(
        self, states: np.ndarray, contexts: List[Dict],
    ) -> np.ndarray:
        """Return (N, 7) float32 sensor predictions.

        Inputs follow the same convention as a list-of-dicts ODSMR call:
        `states[i]` is the (10,) health state for row i, `contexts[i]` is
        the dict with PHASE_TYPE, DTAMB, ALT, MACH, COMMAND.
        """
        if self._closed:
            raise RuntimeError("SurrogateSimulator already closed.")
        if self._validate and not self._validated_once:
            _check_inputs(states, contexts)
            self._validated_once = True
        X = _build_inputs_from_contexts(states, contexts)
        return self._surrogate.predict(X, batch_size=self._batch_size)

    # ------------------------------------------------------------------
    # Lifecycle (matches hpc.SimulatorClient for drop-in compatibility)
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        return not self._closed

    def wait_until_ready(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "SurrogateSimulator":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        cfg = self._surrogate.model
        return (f"SurrogateSimulator(weights={self._weights_path!r}, "
                f"hidden={cfg.hidden}, depth={cfg.depth}, "
                f"batch_size={self._batch_size})")
