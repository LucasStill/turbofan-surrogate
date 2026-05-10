"""Constants for the turbofan surrogate package.

Self-contained: this file deliberately has no project-internal imports
so the standalone release does not depend on the broader
rl_simulator_safran codebase.

Values match the TurboSens / ODSMR conventions used to generate the
training dataset.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# I/O dimensions
# ---------------------------------------------------------------------------
N_COMPONENTS = 10
N_CONTEXTS   = 16
INPUT_DIM    = 17     # state(10) + alt/mach/cmd/dtamb(4) + phase one-hot(3)
OUTPUT_DIM   = 7

# ---------------------------------------------------------------------------
# Phase one-hot ordering (this is the ordering used at training time
# and stored in INPUT_LABELS below; do not change without retraining).
# ---------------------------------------------------------------------------
PHASE_ORDER = ("MTO", "MCL", "CR")

# ---------------------------------------------------------------------------
# Health state labels — same ordering as scenarios.turbosens1.constants
# ---------------------------------------------------------------------------
STATE_LABELS = [
    "deg_CmpBst_s_mapEff_in",
    "deg_CmpBst_s_mapWc_in",
    "deg_CmpFan_s_mapEff_in",
    "deg_CmpFan_s_mapWc_in",
    "deg_CmpH_s_mapEff_in",
    "deg_CmpH_s_mapWc_in",
    "deg_TrbH_s_mapEff_in",
    "deg_TrbH_s_mapWc_in",
    "deg_TrbL_s_mapEff_in",
    "deg_TrbL_s_mapWc_in",
]

# ---------------------------------------------------------------------------
# Sensor names (output column ordering)
# ---------------------------------------------------------------------------
SENSOR_NAMES = [
    "HPC_Tout",      # K
    "HP_Nmech",      # rpm
    "HPC_Tin",       # K
    "LPT_Tin",       # K
    "Fuel_flow",     # kg/s
    "HPC_Pout_st",   # Pa
    "LP_Nmech",      # rpm
]

# ---------------------------------------------------------------------------
# Input vector layout — order of columns in the (n, 17) input matrix
# fed to MLP.apply.
# ---------------------------------------------------------------------------
INPUT_LABELS = [
    "Bst_Eff", "Bst_Wc",
    "Fan_Eff", "Fan_Wc",
    "HPC_Eff", "HPC_Wc",
    "HPT_Eff", "HPT_Wc",
    "LPT_Eff", "LPT_Wc",
    "ALT", "MACH", "COMMAND", "DTAMB",
    "phase_MTO", "phase_MCL", "phase_CR",
]
assert len(INPUT_LABELS) == INPUT_DIM
assert len(STATE_LABELS) == N_COMPONENTS
assert len(SENSOR_NAMES) == OUTPUT_DIM

# ---------------------------------------------------------------------------
# 16 named contexts (operating points). Index 0..15 corresponds to the
# `ci` column the surrogate consumes via build_inputs(...). Phase strings
# match PHASE_ORDER above.
# ---------------------------------------------------------------------------
CONTEXT_NAMES = [
    # MTO (takeoff)
    "MTO_rotation", "MTO_initial",
    # MCL (climb)
    "MCL_low", "MCL_mid", "MCL_upper", "MCL_top",
    # CR (cruise step climb + descent start)
    "CR_01", "CR_02", "CR_03", "CR_04", "CR_05",
    "CR_06", "CR_07", "CR_08", "CR_09", "CR_10",
]
CONTEXT_PHASES = [
    "MTO", "MTO",
    "MCL", "MCL", "MCL", "MCL",
    "CR",  "CR",  "CR",  "CR",  "CR",
    "CR",  "CR",  "CR",  "CR",  "CR",
]
assert len(CONTEXT_NAMES)  == N_CONTEXTS
assert len(CONTEXT_PHASES) == N_CONTEXTS
