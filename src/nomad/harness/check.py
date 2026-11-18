"""``nomad check``: G0 parse -> G1 static -> G2 verify -> G3 lower -> G4 diff -> G5 time."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nomad.devicespec.spec import load_device_from_path
from nomad.diagnostics import Diagnostic, Report, write_report
from nomad.difftest.groups import run_group, time_group
from nomad.graph.catalog import load_catalog
from nomad.graph.plan import PlanDoc
from nomad.graph.schema import GraphDoc
from nomad.graph.static import check_plan, is_supported_group
from nomad.harness.lower import artifact_metrics, find_nvcc, lower_run
from nomad.harness.workspace import (
    admit_snapshot,
    log_event,
    read_state,
    read_task,
    verify_readonly,
    write_state,
)
from nomad.pkir.models import PkirDoc, parse_pkir
from nomad.pkir.static import check_pkir
from nomad.verifier import verify_region
from nomad.yaml_lines import LineMap, YamlLoadError, load_yaml, loc_line, loc_pointer, normalize_loc

GATES = ("G0", "G1", "G2", "G3", "G4", "G5")


class HarnessError(Exception):
    """Exit 2: a harness-owned file is missing or changed; never charged to the budget."""


class BudgetExhausted(Exception):
    """Exit 3."""


def _plan_diagnostics(exc: ValidationError, lines: LineMap) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    for error in exc.errors():
        loc = tuple(error.get("loc", ()))
        code = "P-UNKNOWN-KEY" if error.get("type") == "extra_forbidden" else "P-SCHEMA"
        if code == "P-UNKNOWN-KEY" and loc:
            message = f"unknown key {loc[-1]!r}"
        else:
            message = str(error.get("msg", "invalid value"))
        diags.append(
            Diagnostic(
                code=code,
                message=message,
                file="plan.yaml",
                line=loc_line(lines, normalize_loc(loc)),
                pointer=loc_pointer(loc),
            )
        )
    return diags


def _load_plan(run_dir: Path) -> tuple[PlanDoc | None, LineMap, list[Diagnostic]]:
    try:
        data, lines = load_yaml(run_dir / "plan.yaml")
    except YamlLoadError as exc:
        return None, LineMap(), [Diagnostic("P-SCHEMA", exc.message, "plan.yaml", exc.line, "")]
    try:
        return PlanDoc.model_validate(data), lines, []
    except ValidationError as exc:
        return None, lines, _plan_diagnostics(exc, lines)


def _load_harness_files(run_dir: Path) -> tuple[dict[str, Any], GraphDoc, Any]:
    try:
        task = read_task(run_dir)
        graph_data, _ = load_yaml(run_dir / "graph.yaml")
        graph = GraphDoc.model_validate(graph_data)
        catalog = load_catalog(run_dir / "catalog.yaml")
    except (FileNotFoundError, YamlLoadError, ValidationError) as exc:
        raise HarnessError(f"cannot load harness files: {exc}") from exc
    return task, graph, catalog


def _write_and_log(
    run_dir: Path,
    report: Report,
    exit_code: int,
    hashes: dict[str, str] | None = None,
) -> None:
    write_report(run_dir / "diagnostics" / "latest.json", report)
    log_event(
        run_dir,
        {
            "cmd": "check",
            "submission": report.submission,
            "exit": exit_code,
            "gate": report.gate,
            "codes": [diag.code for diag in report.bounded()],
            "hashes": hashes or {},
            "budget": report.to_dict()["budget"],
            "metrics": report.metrics,
        },
    )


def _reject(
    run_dir: Path,
    state: dict[str, Any],
    gate: str,
    diagnostics: list[Diagnostic],
    *,
    checks_left: int,
    tokens_used: int | None,
    metrics: dict[str, Any] | None = None,
) -> tuple[int, Report]:
    report = Report(
        submission=state["submission"],
        verdict="rejected",
        gate=gate,
        diagnostics=diagnostics,
        checks_left=checks_left,
        tokens_used=tokens_used,
        metrics=metrics,
    )
    write_report(run_dir / "diagnostics" / "latest.json", report)
    write_state(run_dir, state)
    _write_and_log(run_dir, report, 1, state.get("hashes"))
    return 1, report


def _admit(
    run_dir: Path,
    state: dict[str, Any],
    gate: str,
    *,
    checks_left: int,
    tokens_used: int | None,
    metrics: dict[str, Any] | None = None,
) -> tuple[int, Report]:
    report = Report(
        submission=state["submission"],
        verdict="admitted",
        gate=gate,
        checks_left=checks_left,
        tokens_used=tokens_used,
        metrics=metrics,
    )
    write_report(run_dir / "diagnostics" / "latest.json", report)
    state.setdefault("admitted_snapshots", []).append(
        admit_snapshot(run_dir, metrics=metrics, submission=state["submission"]).name
    )
    write_state(run_dir, state)
    _write_and_log(run_dir, report, 0, state.get("hashes"))
    return 0, report


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency
        return False
    return torch.cuda.is_available()


def run_check(run_dir: str | Path, upto: str = "G1") -> tuple[int, Report]:
    """Run the ordered gate stack through ``upto``.

    G0 parse -> G1 static -> G2 verify -> G3 lower -> G4 difftest -> G5 timing. G3 writes CUDA
    source even without a toolchain; G4 and G5 need a CUDA runtime, so a CPU checkout stops at G3
    with a `D-UNAVAILABLE` diagnostic rather than pretending a kernel ran.
    """
    run_dir = Path(run_dir)
    if upto not in GATES:
        raise HarnessError(f"unknown gate {upto!r}; choose one of {', '.join(GATES)}")

    state = read_state(run_dir)
    if state.get("closed"):
        raise HarnessError("run is closed; start a new run")
    problem = verify_readonly(run_dir, state)
    if problem:
        raise HarnessError(problem)
    task, graph, catalog = _load_harness_files(run_dir)
    if task.get("contract") != 0:
        raise HarnessError(f"unsupported task contract {task.get('contract')!r}")
    gates_mode = task.get("gates", "full")
    if gates_mode not in {"full", "tests-only"}:
        raise HarnessError(f"unknown task.gates value {gates_mode!r}")

    budget = task.get("budget") or {}
    limit = budget.get("checks")
    used = int(state.get("checks_used", 0))
    if isinstance(limit, int) and used >= limit:
        raise BudgetExhausted(f"check budget exhausted ({used}/{limit})")
    wall_limit = budget.get("wall_clock_min")
    if isinstance(wall_limit, (int, float)):
        created = datetime.fromisoformat(state["created"])
        elapsed_min = (datetime.now(UTC) - created).total_seconds() / 60.0
        if elapsed_min >= float(wall_limit):
            raise BudgetExhausted(
                f"wall clock budget exhausted ({elapsed_min:.1f}/{wall_limit} min)"
            )

    state["submission"] = int(state.get("submission", 0)) + 1
    state["checks_used"] = used + 1
    checks_left = (limit - state["checks_used"]) if isinstance(limit, int) else 0
    tokens_used = state.get("tokens_used")
    device = load_device_from_path(run_dir / "devicespec.yaml")

    plan, plan_lines, plan_diags = _load_plan(run_dir)

    # G0: parse the plan and every PKIR file named by it.
    if plan_diags:
        return _reject(
            run_dir,
            state,
            "parse",
            plan_diags,
            checks_left=checks_left,
            tokens_used=tokens_used,
        )

    assert plan is not None
    docs: dict[str, PkirDoc] = {}
    g0: list[Diagnostic] = []
    for group in plan.groups:
        path = run_dir / group.pkir
        if not path.is_file():
            g0.append(
                Diagnostic(
                    code="P-SCHEMA",
                    message=f"group {group.id!r} names missing PKIR file {group.pkir!r}",
                    file="plan.yaml",
                    line=loc_line(plan_lines, ("groups", plan.groups.index(group))),
                    pointer=f"/groups/{plan.groups.index(group)}/pkir",
                    group=group.id,
                    hint=f"run `nomad scaffold {group.id}`",
                )
            )
            continue
        doc, diags = parse_pkir(path, display=group.pkir)
        g0.extend(diags)
        if doc is not None:
            docs[group.id] = doc

    referenced = {group.pkir for group in plan.groups}
    for path in sorted((run_dir / "kernels").glob("*.pkir.yaml")):
        relative = str(path.relative_to(run_dir))
        if relative in referenced:
            continue
        _, diags = parse_pkir(path, display=relative)
        g0.extend(diags)

    if g0:
        return _reject(
            run_dir,
            state,
            "parse",
            g0,
            checks_left=checks_left,
            tokens_used=tokens_used,
        )

    if upto == "G0":
        return _admit(run_dir, state, "parse", checks_left=checks_left, tokens_used=tokens_used)

    if gates_mode == "tests-only" and upto in {"G1", "G2"}:
        raise HarnessError("gates: tests-only starts at G3; use --upto G3, G4 or G5")

    if gates_mode == "full":
        # G1: plan legality, then supported PKIR regions.
        g1 = check_plan(plan, graph, catalog, run_dir, plan_lines)
        if not g1:
            for group in plan.groups:
                if not is_supported_group(graph, catalog, group):
                    continue
                doc = docs[group.id]
                _, lines = load_yaml(run_dir / group.pkir)
                g1.extend(check_pkir(doc, device, group, graph, plan, group.pkir, lines))

        if g1:
            return _reject(
                run_dir,
                state,
                "static",
                g1,
                checks_left=checks_left,
                tokens_used=tokens_used,
            )
        if upto == "G1":
            return _admit(
                run_dir, state, "static", checks_left=checks_left, tokens_used=tokens_used
            )

        # G2: slot discipline for every supported region.
        g2: list[Diagnostic] = []
        for group in plan.groups:
            if not is_supported_group(graph, catalog, group):
                continue
            doc = docs[group.id]
            _, lines = load_yaml(run_dir / group.pkir)
            g2.extend(verify_region(doc, group, group.pkir, lines))
        if g2:
            return _reject(
                run_dir, state, "verify", g2, checks_left=checks_left, tokens_used=tokens_used
            )
        if upto == "G2":
            return _admit(
                run_dir, state, "verify", checks_left=checks_left, tokens_used=tokens_used
            )

    # G3: lower every region and write CUDA source (+ PTX when nvcc is available).
    artifacts, g3 = lower_run(
        run_dir,
        plan,
        docs,
        device,
        want_ptx=find_nvcc() is not None,
    )
    if g3:
        return _reject(
            run_dir, state, "lower", g3, checks_left=checks_left, tokens_used=tokens_used
        )
    lower_metrics = artifact_metrics(artifacts, run_dir)
    if upto == "G3":
        return _admit(
            run_dir,
            state,
            "lower",
            checks_left=checks_left,
            tokens_used=tokens_used,
            metrics=lower_metrics,
        )

    if not _cuda_available():
        return _reject(
            run_dir,
            state,
            "difftest",
            [
                Diagnostic(
                    code="D-UNAVAILABLE",
                    message=(
                        "G4 needs a CUDA runtime; the lowered source is available under lowered/"
                    ),
                    file="task.yaml",
                    line=1,
                    pointer="/device",
                    hint="run this gate on the sm86 GPU box, or stop at --upto G3",
                )
            ],
            checks_left=checks_left,
            tokens_used=tokens_used,
            metrics=lower_metrics,
        )

    # G4: build each lowered extension and compare over the task's shape bindings.
    built: dict[str, Any] = {}
    g4: list[Diagnostic] = []
    for group, artifact in artifacts.items():
        try:
            built[group] = artifact.kernel.build()
        except Exception as exc:  # external compiler output; keep the diagnostic bounded
            g4.append(
                Diagnostic(
                    code="D-CRASH",
                    message=f"cannot build lowered {group!r}: {type(exc).__name__}: {exc}"[:300],
                    file=group,
                    line=1,
                    pointer="/kernel",
                    group=group,
                )
            )
    if g4:
        return _reject(
            run_dir,
            state,
            "difftest",
            g4,
            checks_left=checks_left,
            tokens_used=tokens_used,
            metrics=lower_metrics,
        )

    difftest_metrics: dict[str, Any] = {}
    for group, artifact in artifacts.items():
        try:
            results = run_group(
                built[group],
                artifact.kernel.function,
                graph,
                plan,
                plan.group(group),
                task,
            )
        except Exception as exc:
            g4.append(
                Diagnostic(
                    code="D-CRASH",
                    message=f"{type(exc).__name__}: {exc}"[:300],
                    file=plan.group(group).pkir,
                    line=1,
                    pointer="/kernel",
                    group=group,
                )
            )
            continue
        difftest_metrics[group] = [result.__dict__ for result in results]
        for result in results:
            if result.error:
                g4.append(
                    Diagnostic(
                        code="D-CRASH",
                        message=f"{result.shape}: {result.error}"[:300],
                        file=plan.group(group).pkir,
                        line=1,
                        pointer="/body",
                        group=group,
                    )
                )
            elif not result.ok:
                worst = result.worst_index
                g4.append(
                    Diagnostic(
                        code="D-MISMATCH",
                        message=(
                            f"{result.shape}: {result.mismatched} of {result.numel} elements "
                            f"differ; worst |err| {result.max_abs_err:.4g} at {worst}"
                        )[:300],
                        file=plan.group(group).pkir,
                        line=1,
                        pointer="/body",
                        group=group,
                        where={"shape": result.shape, "index": worst},
                    )
                )
    if g4:
        return _reject(
            run_dir,
            state,
            "difftest",
            g4,
            checks_left=checks_left,
            tokens_used=tokens_used,
            metrics={**lower_metrics, "difftest": difftest_metrics},
        )
    if not any(difftest_metrics.values()):
        return _reject(
            run_dir,
            state,
            "difftest",
            [
                Diagnostic(
                    code="D-UNAVAILABLE",
                    message="G4 has no shape bindings for this task",
                    file="task.yaml",
                    line=1,
                    pointer="/shape_sets",
                )
            ],
            checks_left=checks_left,
            tokens_used=tokens_used,
            metrics=lower_metrics,
        )

    g4_metrics = {**lower_metrics, "difftest": difftest_metrics}
    if upto == "G4":
        return _admit(
            run_dir,
            state,
            "difftest",
            checks_left=checks_left,
            tokens_used=tokens_used,
            metrics=g4_metrics,
        )

    # G5: measured timing for each group; never rejects.
    timing: dict[str, Any] = {}
    speedups: list[float] = []
    for group, artifact in artifacts.items():
        try:
            rows = time_group(
                built[group],
                artifact.kernel.function,
                graph,
                plan,
                plan.group(group),
                task,
            )
        except Exception as exc:
            timing[group] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        timing[group] = rows
        speedups.extend(
            float(row["speedup"])
            for row in rows
            if row.get("speedup") is not None and float(row["speedup"]) > 0
        )
    geomean = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups)) if speedups else None
    )
    return _admit(
        run_dir,
        state,
        "timing",
        checks_left=checks_left,
        tokens_used=tokens_used,
        metrics={**g4_metrics, "timing": timing, "geomean_speedup": geomean},
    )
