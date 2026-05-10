"""Turbofan engine sensor surrogate (Flax MLP).

Quick example
-------------
>>> import numpy as np
>>> from turbofan_surrogate import (
...     load_surrogate, build_inputs, CONTEXT_PHASES,
... )
>>> surrogate = load_surrogate("weights/default.pkl")
>>> n = 1000
>>> states = np.random.uniform(-0.05, 0.0, (n, 10)).astype("float32")
>>> ci     = np.random.randint(0, 16, n)
>>> dtamb  = np.full(n, 10.0, dtype="float32")
>>> alt    = np.full(n, 35000.0, dtype="float32")
>>> mach   = np.full(n, 0.78, dtype="float32")
>>> cmd    = np.full(n, 25000.0, dtype="float32")
>>> X      = build_inputs(states, ci, dtamb, alt, mach, cmd,
...                       context_phases=CONTEXT_PHASES)
>>> sensors = surrogate.predict(X)            # (n, 7) float32
"""
from turbofan_surrogate.model import (
    MLP,
    Surrogate,
    load_surrogate,
    save_surrogate,
)
from turbofan_surrogate.inference import build_inputs, phase_to_one_hot
from turbofan_surrogate.simulator import SurrogateSimulator
from turbofan_surrogate.constants import (
    CONTEXT_NAMES,
    CONTEXT_PHASES,
    INPUT_DIM,
    INPUT_LABELS,
    N_COMPONENTS,
    N_CONTEXTS,
    OUTPUT_DIM,
    PHASE_ORDER,
    SENSOR_NAMES,
    STATE_LABELS,
)

__all__ = [
    "MLP",
    "Surrogate",
    "SurrogateSimulator",
    "load_surrogate",
    "save_surrogate",
    "build_inputs",
    "phase_to_one_hot",
    "CONTEXT_NAMES",
    "CONTEXT_PHASES",
    "INPUT_DIM",
    "INPUT_LABELS",
    "N_COMPONENTS",
    "N_CONTEXTS",
    "OUTPUT_DIM",
    "PHASE_ORDER",
    "SENSOR_NAMES",
    "STATE_LABELS",
]

__version__ = "0.1.0"
