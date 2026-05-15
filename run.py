"""P-04 benchmark runner."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from typing import Any, Callable

import numpy as np

from harness import run_multi
from report_format import print_report


def agent_factory_from_spec(spec: str) -> Callable[[np.ndarray, dict[str, Any]], Any]:
    module_name, class_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)

    def factory(X: np.ndarray, params: dict[str, Any]):
        return cls(X, params)

    return factory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Anvil P-04 · PCAM benchmark")
    ap.add_argument("--adapter", required=True,
                    help="module:Class, e.g. adapters.archecho:Engine")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 101, 202, 303, 404])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--noise-levels", type=float, nargs="+",
                    default=[0.6, 0.75, 0.85])
    ap.add_argument("--n-per-level", type=int, default=250)
    ap.add_argument("--n-anisotropy", type=int, default=16)
    ap.add_argument("--format", choices=["table", "json"], default="table")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    print(f"[{time.strftime('%H:%M:%S')}] running {len(args.seeds)} seed(s) ...", file=sys.stderr)
    factory = agent_factory_from_spec(args.adapter)
    t0 = time.monotonic()
    report = run_multi(
        agent_factory=factory,
        seeds=args.seeds,
        K=args.K,
        N=args.N,
        noise_levels=args.noise_levels,
        n_per_level=args.n_per_level,
        n_aniso=args.n_anisotropy,
    )
    total_ms = (time.monotonic() - t0) * 1000.0

    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps(report, indent=2, default=str))

    if args.format == "json":
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report, total_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
