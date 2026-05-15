"""
P-04 self-check.

    python self_check.py --adapter adapters.dummy:DummyAgent --quick
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import Any, Callable

import numpy as np

from harness import run_multi
from report_format import print_advice, print_report


def agent_factory_from_spec(spec: str) -> Callable[[np.ndarray, dict[str, Any]], Any]:
    module_name, class_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    def factory(X: np.ndarray, params: dict[str, Any]):
        return cls(X, params)
    return factory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-04 self-check")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="Smaller K and fewer seeds — fast iteration.")
    args = ap.parse_args(argv)

    if args.quick:
        seeds = [42, 101]
        K, N = 16, 64
        noise_levels = [0.7, 0.8]
        n_per_level = 50
        n_aniso = 5
    else:
        seeds = [42, 101, 202, 303, 404]
        K, N = 16, 64
        noise_levels = [0.5, 0.7, 0.8]
        n_per_level = 250
        n_aniso = 16

    factory = agent_factory_from_spec(args.adapter)
    t0 = time.monotonic()
    report = run_multi(
        agent_factory=factory,
        seeds=seeds,
        K=K, N=N,
        noise_levels=noise_levels,
        n_per_level=n_per_level,
        n_aniso=n_aniso,
    )
    total_ms = (time.monotonic() - t0) * 1000.0

    print_report(report, total_ms)
    print_advice(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
