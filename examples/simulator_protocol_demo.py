"""Drop-in replacement demo: SurrogateSimulator standing in for an ODSMR call.

Given a list of (state, context-dict) pairs, returns 7 sensor predictions
per pair. The dict keys match the OpenDeckSMR FlightCondDeckSMR field
names so existing call sites can be ported by changing one line:

    # before:
    sensors = decksmr_parallel(states, contexts, sensors=..., sim_root=...)

    # after:
    sim     = SurrogateSimulator.from_hf("LucasThil/turbofan-surrogate")
    sensors = sim.simulate(states, contexts)

Runs on any machine with a Python 3.10+ install. Uses GPU if JAX picks
one up (install with `pip install "jax[cuda12]"` for Linux + CUDA 12);
otherwise falls back to CPU automatically.

Usage:
    python examples/simulator_protocol_demo.py
    python examples/simulator_protocol_demo.py --variant large --n 5000
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from turbofan_surrogate import SurrogateSimulator, SENSOR_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id", default="LucasThil/turbofan-surrogate",
        help="HuggingFace model repo id (default: %(default)s).",
    )
    p.add_argument(
        "--variant", default="default",
        choices=("tiny", "small", "default", "large"),
        help="Which weight file to pull (default: %(default)s).",
    )
    p.add_argument(
        "--weights", default=None,
        help="Local path to a surrogate.pkl, overrides --variant.",
    )
    p.add_argument("--n", type=int, default=2000,
                   help="Number of (state, context) pairs to predict.")
    return p.parse_args()


def _make_random_inputs(n: int, rng):
    """Random (states, contexts list) covering the training operating envelope."""
    states = rng.uniform(-0.05, 0.0, (n, 10)).astype("float32")
    states[:, 1::2] = rng.uniform(-0.05, 0.05, (n, 5))      # Wc dims

    phases = rng.choice(["MTO", "MCL", "CR"], size=n,
                        p=[0.125, 0.25, 0.625])
    contexts = []
    for p in phases:
        if p == "MTO":
            contexts.append(dict(PHASE_TYPE="MTO", DTAMB=15.0,
                                 ALT=100.0, MACH=0.0, COMMAND=120000.0))
        elif p == "MCL":
            contexts.append(dict(PHASE_TYPE="MCL", DTAMB=10.0,
                                 ALT=float(rng.uniform(15000, 35000)),
                                 MACH=0.55, COMMAND=110000.0))
        else:
            contexts.append(dict(PHASE_TYPE="CR", DTAMB=10.0,
                                 ALT=float(rng.uniform(33000, 38500)),
                                 MACH=float(rng.uniform(0.78, 0.82)),
                                 COMMAND=float(rng.uniform(20000, 26000))))
    return states, contexts


def main() -> None:
    args = parse_args()

    if args.weights:
        print(f"Loading surrogate from local path: {args.weights}")
        sim = SurrogateSimulator(weights_path=args.weights)
    else:
        print(f"Loading surrogate from HuggingFace: "
              f"{args.repo_id}:weights/{args.variant}.pkl")
        sim = SurrogateSimulator.from_hf(
            repo_id=args.repo_id,
            filename=f"weights/{args.variant}.pkl",
        )
    print(f"  {sim}")

    rng = np.random.default_rng(0)
    states, contexts = _make_random_inputs(args.n, rng)

    # First call compiles the JAX forward; warm up before timing
    _ = sim.simulate(states[:128], contexts[:128])

    t0 = time.perf_counter()
    sensors = sim.simulate(states, contexts)
    elapsed = time.perf_counter() - t0
    print(f"\nPredicted {args.n:,} pairs in {elapsed * 1000:.1f} ms "
          f"({args.n / max(elapsed, 1e-9):,.0f} samples/sec)")
    print(f"sensors shape: {sensors.shape}, dtype: {sensors.dtype}")

    print("\nFirst three predictions:")
    for i in range(min(3, args.n)):
        print(f"  ctx={contexts[i]['PHASE_TYPE']} "
              f"alt={contexts[i]['ALT']:>7.0f} "
              f"mach={contexts[i]['MACH']:.3f}  ->  "
              + ", ".join(f"{n}={sensors[i, j]:.4g}"
                          for j, n in enumerate(SENSOR_NAMES)))

    sim.close()


if __name__ == "__main__":
    main()
