"""Quick inference + timing demo for the turbofan surrogate.

Usage:
    python examples/inference_demo.py --weights weights/default.pkl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from turbofan_surrogate import (
    CONTEXT_NAMES,
    CONTEXT_PHASES,
    SENSOR_NAMES,
    build_inputs,
    load_surrogate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights", default="weights/default.pkl",
        help="Path to a .pkl weights file shipped under weights/.",
    )
    p.add_argument("--n", type=int, default=10_000,
                   help="Total samples to generate / time.")
    p.add_argument("--batch", type=int, default=4096,
                   help="Inference batch size.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.weights} ...")
    surrogate = load_surrogate(args.weights)
    print(f"  hidden={surrogate.model.hidden}, depth={surrogate.model.depth}")

    rng = np.random.default_rng(args.seed)
    n = args.n

    # Sample uniformly within the operating envelope used at training time.
    states = rng.uniform(-0.05, 0.0, (n, 10)).astype("float32")
    states[:, 1::2] = rng.uniform(-0.05, 0.05, (n, 5)).astype("float32")  # Wc dims
    ci    = rng.integers(0, 16, n)
    dtamb = rng.uniform(-15.0, 30.0, n).astype("float32")
    alt   = rng.uniform(0.0,   39_000.0, n).astype("float32")
    mach  = rng.uniform(0.0,   0.84,  n).astype("float32")
    cmd   = rng.uniform(15_000.0, 125_000.0, n).astype("float32")

    X = build_inputs(states, ci, dtamb, alt, mach, cmd)
    print(f"\nInput tensor X: shape={X.shape}, dtype={X.dtype}")

    # Warm up (JAX jit compilation)
    _ = surrogate.predict(X[: args.batch], batch_size=args.batch)

    # Time
    t0 = time.perf_counter()
    Y = surrogate.predict(X, batch_size=args.batch)
    elapsed = time.perf_counter() - t0
    print(f"Predicted {n:,} rows in {elapsed * 1000:.1f} ms "
          f"({n / max(elapsed, 1e-9):,.0f} samples/sec, batch={args.batch})")

    # Show a few representative predictions per phase
    print("\nFirst sample per phase:")
    for phase in ("MTO", "MCL", "CR"):
        idx = next((i for i in range(n) if CONTEXT_PHASES[ci[i]] == phase), None)
        if idx is None:
            continue
        print(f"  ci={ci[idx]:>2} ({CONTEXT_NAMES[ci[idx]]:<14} {phase})  "
              f"alt={alt[idx]:>7.0f} ft  mach={mach[idx]:.3f}")
        for j, name in enumerate(SENSOR_NAMES):
            print(f"    {name:<14}: {Y[idx, j]:.4g}")
        print()


if __name__ == "__main__":
    main()
