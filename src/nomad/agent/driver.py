"""The agent loop around the harness CLI, with a deliberately small file-update protocol.

The agent never writes Python, CUDA or NKI: it returns YAML patches for `plan.yaml` and
`kernels/*.pkir.yaml`, and the harness runs the gate stack. This keeps the same driver usable with
any OpenAI-compatible endpoint, including the local reasoning model used on the laptop.

Response format (one or more blocks):

    ### FILE: kernels/qkv_proj.pkir.yaml
    ```yaml
    pkir: 0
    ...
    ```

`render_prompt` is deterministic and bounded: graph, catalog, plan, kernels and the latest
diagnostics. A missing response, an unsafe path or a non-YAML block is an agent error, never a
silent edit.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from nomad.harness.check import run_check
from nomad.harness.workspace import read_state

_UPDATE = re.compile(
    r"^###\s*FILE:\s*(?P<path>[^\s`]+)\s*$\n```(?:yaml|yml)?\s*\n(?P<body>.*?)^```",
    re.MULTILINE | re.DOTALL,
)
_ALLOWED = re.compile(r"^(plan\.yaml|kernels/[A-Za-z_][A-Za-z0-9_.-]*\.pkir\.yaml)$")


class AgentError(RuntimeError):
    """The model did not produce a usable, safe update."""


@dataclass(frozen=True)
class Update:
    path: str
    text: str


@dataclass(frozen=True)
class AgentRun:
    status: str
    steps: int
    tokens: int | None
    report: dict | None
    files: tuple[str, ...]


def _read(path: Path, *, limit: int = 120_000) -> str:
    text = path.read_text(encoding="utf-8")
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]\n"


def render_prompt(run_dir: str | Path, *, gate: str = "G1") -> str:
    """Build the next prompt from the workspace's read-only inputs and current submission."""
    run_dir = Path(run_dir)
    task = _read(run_dir / "task.yaml")
    graph = _read(run_dir / "graph.yaml")
    catalog = _read(run_dir / "catalog.yaml")
    plan = _read(run_dir / "plan.yaml")
    kernels = []
    for path in sorted((run_dir / "kernels").glob("*.pkir.yaml")):
        kernels.append(f"### FILE: {path.relative_to(run_dir)}\n```yaml\n{_read(path)}```\n")
    latest = run_dir / "diagnostics" / "latest.json"
    diagnostics = _read(latest) if latest.is_file() else "none yet"
    return f"""You are the NOMAD kernel agent. Edit only `plan.yaml` and `kernels/*.pkir.yaml`.
Do not edit `task.yaml`, `graph.yaml`, `devicespec.yaml`, or `catalog.yaml`.
Return complete files, each introduced by `### FILE: <relative path>` and a yaml fence.
Target gate: {gate}. The harness will reject illegal plans with located diagnostics; fix them.

## task.yaml
```yaml
{task}```

## graph.yaml
```yaml
{graph}```

## catalog.yaml
```yaml
{catalog}```

## plan.yaml
```yaml
{plan}```

## kernels
{kernels}
## latest diagnostics
```json
{diagnostics}```
"""


def parse_updates(text: str) -> list[Update]:
    """Parse and validate file blocks. Unsafe or non-YAML updates raise `AgentError`."""
    updates: list[Update] = []
    for match in _UPDATE.finditer(text):
        path = match.group("path").strip()
        if path.startswith("/") or ".." in Path(path).parts or not _ALLOWED.fullmatch(path):
            raise AgentError(f"agent tried to write outside its workspace: {path!r}")
        body = match.group("body")
        try:
            yaml.safe_load(body)
        except yaml.YAMLError as exc:
            raise AgentError(f"agent returned invalid YAML for {path}: {exc}") from exc
        updates.append(Update(path, body if body.endswith("\n") else body + "\n"))
    return updates


def apply_updates(run_dir: str | Path, updates: Iterable[Update]) -> tuple[str, ...]:
    """Write validated updates atomically enough for a single-writer workspace."""
    run_dir = Path(run_dir)
    written: list[str] = []
    for update in updates:
        target = run_dir / update.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.is_relative_to(run_dir):
            raise AgentError(f"agent path escaped the workspace: {update.path!r}")
        target.write_text(update.text, encoding="utf-8")
        written.append(update.path)
    return tuple(written)


def run_agent(
    run_dir: str | Path,
    complete: Callable[[str], str],
    *,
    upto: str = "G1",
    max_steps: int = 8,
    token_counter: Callable[[], int | None] | None = None,
) -> AgentRun:
    """Prompt, apply, gate, repeat. The run ends at admission or `max_steps`."""
    run_dir = Path(run_dir)
    files: list[str] = []
    last: dict | None = None
    for step in range(1, max_steps + 1):
        text = complete(render_prompt(run_dir, gate=upto))
        updates = parse_updates(text)
        if not updates:
            return AgentRun(
                "no-updates",
                step,
                token_counter() if token_counter else None,
                last,
                tuple(files),
            )
        files.extend(apply_updates(run_dir, updates))
        code, report = run_check(run_dir, upto=upto)
        last = report.to_dict()
        if code == 0:
            return AgentRun(
                "admitted",
                step,
                token_counter() if token_counter else None,
                last,
                tuple(files),
            )
    state = read_state(run_dir)
    return AgentRun(
        "max-steps",
        max_steps,
        token_counter() if token_counter else None,
        {
            "verdict": "rejected",
            "diagnostics": (last or {}).get("diagnostics", []),
            "checks_used": state.get("checks_used", 0),
        },
        tuple(files),
    )


class OpenAICompleter:
    """Minimal OpenAI-compatible chat client; no SDK dependency."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 180.0,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        self.url = (
            endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"
        )
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.tokens_used = 0

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the NOMAD kernel agent. Return complete YAML files only, each "
                        "preceded by `### FILE: <relative path>` and a fenced yaml block."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AgentError(f"model request failed: {exc}") from exc
        usage = body.get("usage") or {}
        self.tokens_used += int(usage.get("total_tokens") or 0)
        try:
            content = body["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError(f"unexpected model response: {body}") from exc
        if not content:
            raise AgentError(
                "model returned no content (a reasoning-only response); raise max_tokens or "
                "use a content-first model"
            )
        return str(content)

    def tokens(self) -> int:
        return self.tokens_used
