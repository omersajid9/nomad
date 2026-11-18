"""The agent-facing diagnostic document (agent-contract §7).

One ``Report`` is written to ``diagnostics/latest.json`` per ``nomad check``. Diagnostics are
bounded (five), short (300 characters) and device-free: they name plan/PKIR paths, lines and
DeviceSpec keys, never vendor instructions or compiler output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_DIAGNOSTICS = 5
MAX_MESSAGE = 300


def _short(message: str) -> str:
    message = " ".join(str(message).split())
    if len(message) <= MAX_MESSAGE:
        return message
    return message[: MAX_MESSAGE - 3] + "..."


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    file: str
    line: int = 1
    pointer: str = ""
    group: str | None = None
    where: dict[str, Any] | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "file": self.file,
            "line": int(self.line),
            "pointer": self.pointer,
            "message": _short(self.message),
        }
        if self.group is not None:
            out["group"] = self.group
        if self.where is not None:
            out["where"] = self.where
        if self.hint is not None:
            out["hint"] = _short(self.hint)
        return out


@dataclass
class Report:
    submission: int
    verdict: str
    gate: str | None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    metrics: dict[str, Any] | None = None
    checks_left: int = 0
    tokens_used: int | None = None
    contract: int = 0

    def bounded(self) -> list[Diagnostic]:
        ordered = sorted(
            self.diagnostics,
            key=lambda d: (d.file, d.line, d.pointer, d.code),
        )
        return ordered[:MAX_DIAGNOSTICS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "submission": self.submission,
            "verdict": self.verdict,
            "gate": self.gate,
            "diagnostics": [d.to_dict() for d in self.bounded()],
            "metrics": self.metrics,
            "budget": {
                "checks_left": max(0, self.checks_left),
                "tokens_used": self.tokens_used,
            },
        }

    def summary(self) -> str:
        if self.verdict == "admitted":
            return f"admitted at {self.gate or 'G0'}"
        head = f"rejected at {self.gate or 'unknown'}"
        lines = [head, f"{len(self.bounded())} diagnostic(s)"]
        for diag in self.bounded():
            lines.append(f"  {diag.code} {diag.file}:{diag.line} {diag.message}")
        return "\n".join(lines)


def write_report(path: str | Path, report: Report) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
