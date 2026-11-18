"""G3: lower PKIR regions, write CUDA source, and optionally compile PTX.

The harness owns this step; the agent only edits PKIR. A lowering error is an `L-*` diagnostic
located by the PKIR JSON Pointer the template rejected. Source generation is always available;
compilation to PTX or a loadable extension needs a CUDA toolchain and is reported separately so a
CPU checkout can still inspect the generated CUDA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nomad.devicespec.spec import DeviceSpec, load_device_from_path
from nomad.diagnostics import Diagnostic
from nomad.graph.plan import PlanDoc
from nomad.lowering import LoweredKernel, LoweringError, lower
from nomad.pkir.models import PkirDoc, parse_pkir
from nomad.yaml_lines import LineMap, load_yaml, loc_line

LOWERED_DIR = "lowered"
_POINTER_KEYS = {"async": "async_", "for": "for_", "copy": "copy_", "matmul": "matmul"}


@dataclass(frozen=True)
class LoweredArtifact:
    group: str
    kernel: LoweredKernel
    source_path: Path
    meta_path: Path
    ptx_path: Path | None = None
    compiled: bool = False
    compile_error: str | None = None
    ptx_error: str | None = None


def _pointer_loc(pointer: str) -> tuple[Any, ...]:
    if not pointer:
        return ()
    parts: list[Any] = []
    for raw in pointer.strip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        parts.append(int(part) if part.isdigit() else _POINTER_KEYS.get(part, part))
    return tuple(parts)


def _lower_diag(
    exc: LoweringError,
    group: str,
    display: str,
    lines: LineMap,
) -> Diagnostic:
    path = _pointer_loc(exc.pointer)
    return Diagnostic(
        code=exc.code,
        message=exc.message,
        file=display,
        line=loc_line(lines, path),
        pointer=exc.pointer,
        group=group,
    )


def find_nvcc() -> str | None:
    """Locate nvcc on PATH or under CUDA_HOME/CUDA_PATH."""
    found = shutil.which("nvcc")
    if found:
        return found
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if root:
            candidate = Path(root) / "bin" / "nvcc"
            if candidate.is_file():
                return str(candidate)
    return None


def compile_ptx(kernel: LoweredKernel, output: Path, *, timeout: int = 180) -> Path:
    """Compile the device-only half of a lowered kernel to PTX with `nvcc -ptx`."""
    nvcc = find_nvcc()
    if nvcc is None:
        raise LoweringError(
            "L-TOOLCHAIN",
            "",
            "nvcc not found on PATH or under CUDA_HOME; cannot emit PTX on this machine",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.with_suffix(".ptx.cu")
    source.write_text(kernel.device_source, encoding="utf-8")
    target_arch = kernel.cuda_cflags[-1].rsplit("_", 1)[-1] if kernel.cuda_cflags else "86"
    command = [
        nvcc,
        "-ptx",
        f"-arch=sm_{target_arch}",
        "-O3",
        "-std=c++17",
        str(source),
        "-o",
        str(output),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LoweringError("L-COMPILE", "", f"nvcc failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        message = detail[-1] if detail else f"nvcc exited {proc.returncode}"
        raise LoweringError("L-COMPILE", "", message[:300])
    return output


def lower_run(
    run_dir: str | Path,
    plan: Any,
    docs: dict[str, PkirDoc],
    device: DeviceSpec,
    *,
    want_ptx: bool = False,
    build: bool = False,
    group_id: str | None = None,
) -> tuple[dict[str, LoweredArtifact], list[Diagnostic]]:
    """Lower every group and write `lowered/GROUP.cu` (+ `.ptx` when requested)."""
    run_dir = Path(run_dir)
    root = run_dir / LOWERED_DIR
    root.mkdir(exist_ok=True)
    artifacts: dict[str, LoweredArtifact] = {}
    diags: list[Diagnostic] = []

    for group in plan.groups:
        if group_id is not None and group.id != group_id:
            continue
        doc = docs[group.id]
        display = group.pkir
        _, lines = load_yaml(run_dir / group.pkir)
        try:
            kernel = lower(doc, device)
        except LoweringError as exc:
            diags.append(_lower_diag(exc, group.id, display, lines))
            continue

        source_path = root / f"{group.id}.cu"
        meta_path = root / f"{group.id}.json"
        source_path.write_text(kernel.source, encoding="utf-8")

        ptx_path: Path | None = None
        ptx_error: str | None = None
        if want_ptx:
            try:
                ptx_path = compile_ptx(kernel, root / f"{group.id}.ptx")
            except LoweringError as exc:
                ptx_error = exc.message
                diags.append(_lower_diag(exc, group.id, display, lines))

        compiled = False
        compile_error: str | None = None
        if build:
            try:
                kernel.build()
                compiled = True
            except Exception as exc:  # compilation is external; keep the message bounded
                compile_error = f"{type(exc).__name__}: {exc}"[:300]

        meta = {
            "group": group.id,
            "kernel": kernel.function,
            "module": kernel.name,
            "source": str(source_path.relative_to(run_dir)),
            "ptx": str(ptx_path.relative_to(run_dir)) if ptx_path is not None else None,
            "compiled": compiled,
            "compile_error": compile_error,
            "ptx_error": ptx_error,
            "cuda_cflags": list(kernel.cuda_cflags),
            "shared_bytes": kernel.region.shared_bytes,
            "threads": kernel.region.threads,
            "tile": list(kernel.region.tile),
            "warps": list(kernel.region.warps),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        artifacts[group.id] = LoweredArtifact(
            group=group.id,
            kernel=kernel,
            source_path=source_path,
            meta_path=meta_path,
            ptx_path=ptx_path,
            compiled=compiled,
            compile_error=compile_error,
            ptx_error=ptx_error,
        )

    return artifacts, diags


def artifact_metrics(artifacts: dict[str, LoweredArtifact], run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    return {
        "lowered": {
            group: {
                "source": str(art.source_path.relative_to(run_dir)),
                "ptx": str(art.ptx_path.relative_to(run_dir)) if art.ptx_path else None,
                "compiled": art.compiled,
                "shared_bytes": art.kernel.region.shared_bytes,
                "threads": art.kernel.region.threads,
                "compile_error": art.compile_error,
                "ptx_error": art.ptx_error,
            }
            for group, art in artifacts.items()
        }
    }


def lower_workspace(
    run_dir: str | Path,
    *,
    want_ptx: bool = False,
    group_id: str | None = None,
) -> tuple[dict[str, LoweredArtifact], list[Diagnostic], dict[str, Any]]:
    """Load a run's plan and PKIR, lower it, and return artifacts, diagnostics and metrics."""
    run_dir = Path(run_dir)
    plan_data, _ = load_yaml(run_dir / "plan.yaml")
    plan = PlanDoc.model_validate(plan_data)
    if group_id is not None and group_id not in {group.id for group in plan.groups}:
        raise ValueError(f"unknown group {group_id!r}")
    docs: dict[str, PkirDoc] = {}
    for group in plan.groups:
        if group_id is not None and group.id != group_id:
            continue
        doc, diags = parse_pkir(run_dir / group.pkir, display=group.pkir)
        if diags or doc is None:
            return {}, diags, {}
        docs[group.id] = doc
    device = load_device_from_path(run_dir / "devicespec.yaml")
    artifacts, diags = lower_run(
        run_dir,
        plan,
        docs,
        device,
        want_ptx=want_ptx,
        group_id=group_id,
    )
    return artifacts, diags, artifact_metrics(artifacts, run_dir)
