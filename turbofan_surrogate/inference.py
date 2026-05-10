"""Helpers for building the 17D input vector the surrogate consumes."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from turbofan_surrogate.constants import (
    CONTEXT_PHASES,
    INPUT_DIM,
    PHASE_ORDER,
)


def phase_to_one_hot(phases: np.ndarray) -> np.ndarray:
    """Convert a (n,) array of phase strings to (n, 3) one-hot in PHASE_ORDER."""
    out = np.zeros((len(phases), len(PHASE_ORDER)), dtype=np.float32)
    for j, p in enumerate(PHASE_ORDER):
        out[:, j] = (phases == p).astype(np.float32)
    return out


def build_inputs(
    states: np.ndarray,
    ci: np.ndarray,
    dtamb: np.ndarray,
    alt: np.ndarray,
    mach: np.ndarray,
    cmd: np.ndarray,
    *,
    context_phases: Sequence[str] = CONTEXT_PHASES,
) -> np.ndarray:
    """Assemble the (n, 17) input matrix expected by the surrogate.

    Parameters
    ----------
    states : (n, 10) float32
        Health state vectors. Eff channels (even indices) are in [-0.05, 0];
        Wc channels (odd indices) are in [-0.05, 0.03] or [-0.05, 0.05].
        See STATE_LABELS for the exact ordering.
    ci : (n,) int
        Context indices in [0, 15]. See CONTEXT_NAMES / CONTEXT_PHASES.
    dtamb : (n,) float32   Ambient temperature deviation from ISA, in K.
    alt   : (n,) float32   Altitude in feet.
    mach  : (n,) float32   Mach number.
    cmd   : (n,) float32   Thrust command in lbf (used only for CR phase).
    context_phases : list-like
        Mapping ci -> phase string. Defaults to the standard 16-context
        TurboSens layout. Override only if you trained with a different one.

    Returns
    -------
    X : (n, 17) float32
        Layout: [10 state, ALT, MACH, COMMAND, DTAMB, 3 phase one-hot].
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError(f"states must be (n, 10), got {states.shape}")

    ci    = np.asarray(ci)
    dtamb = np.asarray(dtamb, dtype=np.float32)
    alt   = np.asarray(alt,   dtype=np.float32)
    mach  = np.asarray(mach,  dtype=np.float32)
    cmd   = np.asarray(cmd,   dtype=np.float32)

    phases   = np.array([context_phases[int(c)] for c in ci])
    phase_oh = phase_to_one_hot(phases)
    cont     = np.stack([alt, mach, cmd, dtamb], axis=1).astype(np.float32)
    X        = np.concatenate([states, cont, phase_oh], axis=1).astype(np.float32)

    assert X.shape[1] == INPUT_DIM, (X.shape, INPUT_DIM)
    return X
