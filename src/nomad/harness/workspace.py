"""Run workspace lifecycle (agent-contract §2, §8)."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from nomad.devicespec.spec import DeviceSpec, load_device
from nomad.graph.catalog import DEFAULT_CATALOG
from nomad.graph.ingest import graph_from_export, load_export
from nomad.graph.plan import PlanDoc
from nomad.graph.schema import GraphDoc
from nomad.harness.templates import scaffold_group
from nomad.reference.block import TASK_SHAPES, graph_attention_block, graph_qkv_proj
from nomad.yaml_lines import load_yaml

READ_ONLY = ["task.yaml", "devicespec.yaml", "graph.yaml", "catalog.yaml"]


def resolve_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    if not path.is_absolute() and path.parent == Path("."):
        return Path("results") / "runs" / path
    return path


def _dump(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _builtin_graph(task: str) -> dict[str, Any]:
    if task == "qkv_proj":
        return graph_qkv_proj()
    if task == "attention_block":
        return graph_attention_block()
    raise ValueError(f"unknown task {task!r}; expected qkv_proj or attention_block")


def _plan_unfused(graph: GraphDoc) -> dict[str, Any]:
    return {
        "plan": 0,
        "graph": graph.graph,
        "groups": [
            {
                "id": op.id,
                "pattern": "single",
                "ops": [op.id],
                "pkir": f"kernels/{op.id}.pkir.yaml",
            }
            for op in graph.ops
        ],
        "layouts": {},
        "rewrites": [],
    }


def _task_doc(task: str, mode: str = "graph", start: str = "unfused") -> dict[str, Any]:
    shape_name = "block_v0" if task == "attention_block" else "qkv_v0"
    return {
        "contract": 0,
        "task": task,
        "device": "sm86",
        "shapes": shape_name,
        "shape_sets": TASK_SHAPES,
        "mode": mode,
        "gates": "full",
        "start": start,
        "seed": 0,
        "budget": {"checks": 40, "tokens": None, "wall_clock_min": 120},
    }


def hash_readonly(run_dir: str | Path, mode: str = "graph") -> dict[str, str]:
    run_dir = Path(run_dir)
    names = list(READ_ONLY)
    if mode == "kernel":
        names.append("plan.yaml")
    hashes: dict[str, str] = {}
    for name in names:
        path = run_dir / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def read_task(run_dir: str | Path) -> dict[str, Any]:
    data, _ = load_yaml(Path(run_dir) / "task.yaml")
    return data


def read_state(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def write_state(run_dir: str | Path, state: dict[str, Any]) -> None:
    (Path(run_dir) / "run.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def log_event(run_dir: str | Path, event: dict[str, Any]) -> None:
    event = {"ts": datetime.now(UTC).isoformat(), **event}
    with (Path(run_dir) / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def verify_readonly(run_dir: str | Path, state: dict[str, Any]) -> str | None:
    current = hash_readonly(run_dir, state.get("mode", "graph"))
    expected = state.get("hashes", {})
    for name, digest in expected.items():
        if current.get(name) != digest:
            return f"read-only file changed: {name}"
    return None


def admit_snapshot(
    run_dir: str | Path,
    metrics: dict[str, Any] | None = None,
    submission: int | None = None,
) -> Path:
    run_dir = Path(run_dir)
    index_path = run_dir / "snapshots" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    number = len(index) + 1
    target = run_dir / "snapshots" / f"{number:03d}"
    target.mkdir(parents=True, exist_ok=False)
    for name in ["plan.yaml", "task.yaml", "devicespec.yaml", "graph.yaml", "catalog.yaml"]:
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, target / name)
    if (run_dir / "kernels").is_dir():
        shutil.copytree(run_dir / "kernels", target / "kernels")
    if (run_dir / "lowered").is_dir():
        shutil.copytree(run_dir / "lowered", target / "lowered")
    latest = run_dir / "diagnostics" / "latest.json"
    if latest.is_file():
        (target / "diagnostics").mkdir(exist_ok=True)
        shutil.copy2(latest, target / "diagnostics" / "latest.json")
    if submission is None:
        submission = read_state(run_dir).get("submission", 0)
    index.append(
        {
            "snapshot": f"snapshots/{number:03d}",
            "submission": submission,
            "metrics": metrics,
        }
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return target


def init_run(
    task: str,
    run_dir: str | Path,
    *,
    graph_path: str | Path | None = None,
    export_path: str | Path | None = None,
    mode: str = "graph",
    start: str = "scratch",
) -> Path:
    """Create a run workspace. ``init`` is harness-only; the agent never calls it."""

    if mode == "kernel" and start in {"unfused", "scratch"}:
        raise ValueError("kernel mode requires start: <plan path or run dir>")

    if graph_path is not None:
        data, _ = load_yaml(graph_path)
        graph = GraphDoc.model_validate(data)
        graph_doc = graph.model_dump(by_alias=True, exclude_none=True)
        task = graph.graph
    elif export_path is not None:
        graph_doc = graph_from_export(load_export(export_path), name=task)
        graph = GraphDoc.model_validate(graph_doc)
        task = graph.graph
    else:
        graph_doc = _builtin_graph(task)
        graph = GraphDoc.model_validate(graph_doc)

    run_dir = resolve_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "kernels").mkdir()
    (run_dir / "diagnostics").mkdir()
    (run_dir / "snapshots").mkdir()

    task_doc = _task_doc(task, mode=mode, start=start)
    _dump(run_dir / "task.yaml", task_doc)
    device: DeviceSpec = load_device(task_doc["device"])
    _dump(run_dir / "devicespec.yaml", device.model_dump(by_alias=True))
    _dump(run_dir / "graph.yaml", graph_doc)
    shutil.copy2(DEFAULT_CATALOG, run_dir / "catalog.yaml")

    template = True
    if start == "unfused":
        plan_doc = _plan_unfused(graph)
    elif start == "scratch":
        plan_doc = _plan_unfused(graph)
        template = False
    else:
        source = Path(start)
        plan_path = source / "plan.yaml" if source.is_dir() else source
        data, _ = load_yaml(plan_path)
        plan_doc = PlanDoc.model_validate(data).model_dump()
        if plan_doc["graph"] != graph.graph:
            raise ValueError(f"start plan targets graph {plan_doc['graph']!r}, not {graph.graph!r}")
        if source.is_dir() and (source / "kernels").is_dir():
            shutil.copytree(source / "kernels", run_dir / "kernels", dirs_exist_ok=True)
        template = False
    _dump(run_dir / "plan.yaml", plan_doc)
    plan = PlanDoc.model_validate(plan_doc)
    for group in plan.groups:
        if not (run_dir / group.pkir).is_file():
            scaffold_group(run_dir, graph, plan, group, device.name, template=template)

    state = {
        "run_id": run_dir.name,
        "task": task,
        "mode": mode,
        "start": start,
        "created": datetime.now(UTC).isoformat(),
        "closed": False,
        "submission": 0,
        "checks_used": 0,
        "tokens_used": None,
        "admitted_snapshots": [],
        "hashes": hash_readonly(run_dir, mode),
    }
    write_state(run_dir, state)
    (run_dir / "log.jsonl").touch()
    log_event(run_dir, {"cmd": "init", "task": task, "exit": 0})
    return run_dir


def submit(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    run_dir = Path(run_dir)
    state = read_state(run_dir)
    index_path = run_dir / "snapshots" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    admitted = [entry for entry in index if entry.get("metrics") is not None]
    if not admitted:
        admitted = index  # v0 G1 has no metrics; the latest admitted snapshot is the fallback
    chosen = admitted[-1] if admitted else None
    metrics = chosen.get("metrics") if chosen else None
    if metrics and "timing" in metrics:
        note = "admitted with measured G5 timing"
    elif metrics and "difftest" in metrics:
        note = "admitted at G4; G5 timing not recorded"
    elif metrics and "lowered" in metrics:
        note = "admitted at G3; difftest and timing not recorded"
    else:
        note = "admitted at G1; lowering and timing not recorded"
    result = {
        "status": "admitted" if chosen else "failed",
        "snapshot": chosen["snapshot"] if chosen else None,
        "submission": state.get("submission", 0),
        "checks_used": state.get("checks_used", 0),
        "metrics": metrics,
        "note": note,
    }
    path = run_dir / "result.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    state["closed"] = True
    write_state(run_dir, state)
    log_event(run_dir, {"cmd": "submit", "exit": 0, "result": result})
    return path, result
