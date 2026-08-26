"""End-to-end driver for the Paper 1 pipeline.

Usage
-----
    python scripts/run_pipeline.py --config configs/default.yaml

Stages executed in order:
  1. Load + harmonize raw datasets (§1)
  2. Compute per-cell raw scores P/D/Pr/Ph (§2.1–§2.4)
  3. Assemble trajectories, fit min-max bounds on training split (§2.5, §1.4 Step 4)
  4. Register provenance manifest (§3)
  5. Validation: correlations, MOFA+, coherence (§4)
  6. Benchmark: Tasks A and B, 2D vs 4D, with permutation null (§5)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(f"[pipeline] loaded config: {args.config}")
    print(f"[pipeline] artifacts → {args.out_dir}")
    print(f"[pipeline] seed = {cfg.get('seed', 42)}")

    # TODO: wire up the six stages once the data/score modules are implemented.
    raise SystemExit(
        "Pipeline driver is a scaffold. Implement stages by stitching "
        "stem_dt.{data, harmonize, scores, provenance, validation, benchmark}."
    )


if __name__ == "__main__":
    main()
