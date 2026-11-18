"""One end-to-end pass with the components landed so far.

init -> (optional rewrite choice) -> ordered gates through --upto -> cost -> submit, plus an
optional CPU reference/difftest self-check. ``--upto G3`` writes CUDA source; ``--upto G5`` also
runs G4 difftest and G5 timing on the CUDA box.

    uv run python experiments/single_pass.py --task qkv_proj --run results/runs/pass-qkv
    uv run python experiments/single_pass.py --task qkv_proj --run results/runs/pass-qkv --upto G3
    uv run python experiments/single_pass.py --task attention_block --run results/runs/pass-block
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import yaml

from nomad.cost.model import estimate
from nomad.difftest.harness import run as run_difftest
from nomad.graph.catalog import load_catalog
from nomad.graph.rewrites import feasible_sequences
from nomad.graph.schema import load_graph
from nomad.harness.check import HarnessError, run_check
from nomad.harness.workspace import init_run, submit
from nomad.reference.block import BlockShape, block


def _choose_rewrites(run_dir: Path) -> list[str]:
    graph, _ = load_graph(run_dir / "graph.yaml")
    catalog = load_catalog(run_dir / "catalog.yaml")
    sequences = feasible_sequences(graph, catalog)
    if not sequences:
        return []
    chosen = sequences[-1]  # longest feasible sequence (ties are alphabetical)
    plan = yaml.safe_load((run_dir / "plan.yaml").read_text())
    plan["rewrites"] = [{"rule": rule} for rule in chosen]
    (run_dir / "plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return list(chosen)


def _difftest_self_check() -> dict[str, bool]:
    tiny = (
        BlockShape(batch=2, seq=8, d_model=32, n_heads=4),
        BlockShape(batch=1, seq=5, d_model=32, n_heads=2, causal=True),
    )
    results = {}
    for name, candidate in (
        ("eager", block),
        ("sdpa", partial(block, attention="sdpa")),
    ):
        rows = run_difftest(candidate, shapes=tiny, device="cpu", dtype=torch.float32)
        results[name] = all(row.ok for row in rows)
    return results


def run_pass(
    task: str,
    run_dir: str | Path,
    *,
    start: str = "unfused",
    rewrite_mode: str = "none",
    difftest: bool = False,
    upto: str = "G1",
) -> dict:
    run_dir = init_run(task, run_dir, start=start)
    rewrites = _choose_rewrites(run_dir) if rewrite_mode == "auto" else []
    code, report = run_check(run_dir, upto=upto)
    cost = estimate(run_dir) if code == 0 else {"status": "skipped-after-rejection"}
    result = {
        "run": str(run_dir),
        "task": task,
        "start": start,
        "rewrites": rewrites,
        "check": {"exit": code, "verdict": report.verdict, "gate": report.gate},
        "diagnostics": [d.to_dict() for d in report.bounded()],
        "cost": cost,
        "metrics": report.metrics,
        "difftest": _difftest_self_check() if difftest else None,
    }
    if code == 0:
        _, submitted = submit(run_dir)
        result["result"] = submitted
    else:
        result["result"] = {"status": "failed", "note": "no admitted snapshot"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="qkv_proj", choices=["qkv_proj", "attention_block"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--start", default="unfused", choices=["unfused", "scratch"])
    parser.add_argument("--rewrites", default="none", choices=["none", "auto"])
    parser.add_argument("--difftest", action="store_true")
    parser.add_argument("--upto", default="G1", choices=["G0", "G1", "G2", "G3", "G4", "G5"])
    args = parser.parse_args()
    try:
        result = run_pass(
            args.task,
            args.run,
            start=args.start,
            rewrite_mode=args.rewrites,
            difftest=args.difftest,
            upto=args.upto,
        )
    except HarnessError as exc:
        print(f"harness error: {exc}")
        return 2
    print(json.dumps(result, indent=2))
    return int(result["check"]["exit"])


if __name__ == "__main__":
    raise SystemExit(main())
