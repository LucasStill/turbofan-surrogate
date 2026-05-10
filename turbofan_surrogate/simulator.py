"""SurrogateSimulator — drop-in replacement for an ODSMR call site.

Provides a `simulate(states, contexts) -> (N, 7)` method matching the
contract of any thermodynamic engine simulator that takes a list of
(state, context) pairs and returns sensor observations. Users replacing
ODSMR in an existing pipeline can swap their simulator object for an
instance of this class without changing any of the calling code.

Example
-------
    from turbofan_surrogate import SurrogateSimulator

    # Pull the recommended weights from HuggingFace on first use.
    sim = SurrogateSimulator.from_hf("LucasThil/turbofan-surrogate")

    states = ...      # (N, 10) float32 health states
    contexts = [
        {"PHASE_TYPE": "CR", "DTAMB": 10.0, "ALT": 35000.0,
         "MACH": 0.78, "COMMAND": 25000.0},
        ...                                # one dict per row
    ]
    sensors = sim.simulate(states, contexts)  # (N, 7) float32

The dict keys (`PHASE_TYPE`, `DTAMB`, `ALT`, `MACH`, `COMMAND`) match
the OpenDeckSMR `FlightCondDeckSMR` field names so anyone migrating
from ODSMR can pass their existing context dicts unchanged.

Pure-Python class with no dependency on the broader rl_simulator_safran
codebase — works wherever the standalone `turbofan_surrogate` package
is installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from turbofan_surrogate.constants import OUTPUT_DIM, PHASE_ORDER
from turbofan_surrogate.inference import phase_to_one_hot
from turbofan_surrogate.model import Surrogate, load_surrogate


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


class SurrogateSimulator:
    """Stateful, callable surrogate exposing the ODSMR-style protocol.

    Parameters
    ----------
    weights_path:
        Local path to a `surrogate.pkl` (mutually exclusive with `from_hf`).
    batch_size:
        Inference batch size for the underlying JAX forward.
    """

    def __init__(self, weights_path: str, batch_size: int = 8192) -> None:
        self._weights_path: str = str(Path(weights_path).resolve())
        self._batch_size:   int = batch_size
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
        """Pull weights from HuggingFace and instantiate.

        Requires `huggingface_hub` to be installed. Public repos do not
        require an HF token. The downloaded file is cached under
        $HF_HOME (default `~/.cache/huggingface`); subsequent calls reuse it.
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
        X = _build_inputs_from_contexts(states, contexts)
        return self._surrogate.predict(X, batch_size=self._batch_size)

    # ------------------------------------------------------------------
    # Lifecycle (matches SimulatorClient for drop-in compatibility)
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
