"""``nomad`` command line: the agent's only interface to the harness (agent-contract §5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nomad.agent.driver import (
    OPENROUTER_ENDPOINT,
    OPENROUTER_MODEL,
    AgentError,
    OpenAICompleter,
    OpenRouterCompleter,
    run_agent,
)
from nomad.cost.model import estimate
from nomad.devicespec.spec import load_device_from_path
from nomad.graph.catalog import load_catalog
from nomad.graph.plan import load_plan
from nomad.graph.rewrites import apply_rule, feasible_sequences, initial_facts
from nomad.graph.schema import load_graph
from nomad.harness.check import BudgetExhausted, HarnessError, run_check
from nomad.harness.explain import EXPLAIN
from nomad.harness.lower import lower_workspace
from nomad.harness.templates import scaffold_group
from nomad.harness.workspace import (
    init_run,
    log_event,
    submit,
)
from nomad.yaml_lines import YamlLoadError


def _print_error(message: str) -> None:
    print(f"nomad: {message}", file=sys.stderr)


def _cmd_init(args: argparse.Namespace) -> int:
    run_dir = init_run(
        args.task,
        args.run_dir,
        graph_path=args.graph,
        export_path=args.export,
        mode=args.mode,
        start=args.start,
    )
    print(run_dir)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        code, report = run_check(args.run_dir, upto=args.upto)
    except BudgetExhausted as exc:
        _print_error(str(exc))
        try:
            log_event(args.run_dir, {"cmd": "check", "exit": 3, "error": str(exc)})
        except OSError:
            pass
        return 3
    except HarnessError as exc:
        _print_error(str(exc))
        try:
            log_event(args.run_dir, {"cmd": "check", "exit": 2, "error": str(exc)})
        except OSError:
            pass
        return 2
    print(report.summary())
    print(f"report: {Path(args.run_dir) / 'diagnostics' / 'latest.json'}")
    return code


def _cmd_cost(args: argparse.Namespace) -> int:
    try:
        result = estimate(args.run_dir, group_id=args.group)
    except KeyError as exc:
        _print_error(str(exc))
        return 2
    except (OSError, ValueError) as exc:
        _print_error(f"cannot estimate cost: {exc}")
        return 2
    print(json.dumps(result, indent=2))
    log_event(args.run_dir, {"cmd": "cost", "group": args.group, "exit": 0})
    return 0


def _cmd_scaffold(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    graph, _ = load_graph(run_dir / "graph.yaml")
    plan, _ = load_plan(run_dir / "plan.yaml")
    device = load_device_from_path(run_dir / "devicespec.yaml")
    try:
        group = plan.group(args.group)
    except KeyError:
        _print_error(f"unknown group {args.group!r}")
        return 2
    path = scaffold_group(run_dir, graph, plan, group, device.name)
    log_event(run_dir, {"cmd": "scaffold", "group": args.group, "exit": 0})
    print(path)
    return 0


def _cmd_explore(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    graph, _ = load_graph(run_dir / "graph.yaml")
    catalog = load_catalog(run_dir / "catalog.yaml")
    facts = initial_facts(graph)
    applicable = [rule.id for rule in catalog.rewrites if apply_rule(facts, rule.id)[0] is not None]
    sequences = feasible_sequences(graph, catalog, max_depth=args.depth)
    result = {
        "rules": [rule.id for rule in catalog.rewrites],
        "applicable": applicable,
        "sequences": [list(sequence) for sequence in sequences],
        "count": len(sequences),
    }
    (run_dir / "exploration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    log_event(run_dir, {"cmd": "explore", "exit": 0, "count": result["count"]})
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    text = EXPLAIN.get(args.code)
    if text is None:
        _print_error(f"unknown diagnostic code {args.code!r}")
        return 2
    print(f"{args.code}: {text}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    path, result = submit(args.run_dir)
    print(json.dumps(result, indent=2))
    print(f"result: {path}")
    return 0


def _cmd_lower(args: argparse.Namespace) -> int:
    try:
        _, diagnostics, metrics = lower_workspace(
            args.run_dir,
            want_ptx=args.ptx,
            group_id=args.group,
        )
    except (OSError, ValueError) as exc:
        _print_error(str(exc))
        return 2
    if diagnostics:
        print(
            json.dumps(
                {"diagnostics": [diag.to_dict() for diag in diagnostics[:5]]},
                indent=2,
            )
        )
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    try:
        if args.endpoint == OPENROUTER_ENDPOINT:
            completer = OpenRouterCompleter(
                args.endpoint,
                args.model,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            completer = OpenAICompleter(
                args.endpoint,
                args.model,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
    except AgentError as exc:
        _print_error(str(exc))
        return 2
    try:
        result = run_agent(
            args.run_dir,
            completer,
            upto=args.upto,
            max_steps=args.max_steps,
            token_counter=completer.tokens,
        )
    except AgentError as exc:
        _print_error(str(exc))
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "steps": result.steps,
                "tokens": result.tokens,
                "files": list(result.files),
                "report": result.report,
            },
            indent=2,
        )
    )
    return 0 if result.status == "admitted" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nomad", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    agent = sub.add_parser("agent", help="drive an OpenAI-compatible model against a workspace")
    agent.add_argument("run_dir")
    agent.add_argument(
        "--endpoint",
        default=OPENROUTER_ENDPOINT,
        help=f"OpenAI-compatible base URL (default: {OPENROUTER_ENDPOINT})",
    )
    agent.add_argument(
        "--model",
        default=OPENROUTER_MODEL,
        help=f"model id (default: {OPENROUTER_MODEL})",
    )
    agent.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter key; defaults to OPENROUTER_API_KEY for the OpenRouter endpoint",
    )
    agent.add_argument("--upto", choices=["G0", "G1", "G2", "G3", "G4", "G5"], default="G1")
    agent.add_argument("--max-steps", type=int, default=8)
    agent.add_argument("--max-tokens", type=int, default=4096)
    agent.add_argument("--temperature", type=float, default=0.0)
    agent.set_defaults(func=_cmd_agent)

    init = sub.add_parser("init", help="create a run workspace (harness-only)")
    init.add_argument("task", help="qkv_proj, attention_block, or a graph name with --graph")
    init.add_argument("run_dir", help="run directory, or a name under results/runs/")
    init.add_argument("--graph", help="path to a graph.yaml (overrides the built-in task)")
    init.add_argument("--from-export", dest="export", help="torch.export .py factory or .pt2")
    init.add_argument("--mode", choices=["graph", "kernel"], default="graph")
    init.add_argument("--start", default="scratch", help="scratch, unfused, or a plan/run path")
    init.set_defaults(func=_cmd_init)

    check = sub.add_parser("check", help="run the ordered gates and write diagnostics/latest.json")
    check.add_argument("run_dir")
    check.add_argument("--upto", choices=["G0", "G1", "G2", "G3", "G4", "G5"], default="G1")
    check.set_defaults(func=_cmd_check)

    lower = sub.add_parser("lower", help="lower PKIR to lowered/GROUP.cu (and .ptx with --ptx)")
    lower.add_argument("run_dir")
    lower.add_argument("--group")
    lower.add_argument("--ptx", action="store_true", help="compile device code with nvcc -ptx")
    lower.set_defaults(func=_cmd_lower)

    cost = sub.add_parser("cost", help="analytic cost estimate; never rejects")
    cost.add_argument("run_dir")
    cost.add_argument("--group")
    cost.set_defaults(func=_cmd_cost)

    scaffold = sub.add_parser("scaffold", help="(re)write kernels/GROUP.pkir.yaml")
    scaffold.add_argument("run_dir")
    scaffold.add_argument("group")
    scaffold.set_defaults(func=_cmd_scaffold)

    explore = sub.add_parser("explore", help="enumerate feasible rewrite subsets and orders")
    explore.add_argument("run_dir")
    explore.add_argument("--depth", type=int, default=None)
    explore.set_defaults(func=_cmd_explore)

    explain = sub.add_parser("explain", help="print documentation for a diagnostic code")
    explain.add_argument("code")
    explain.set_defaults(func=_cmd_explain)

    submit = sub.add_parser("submit", help="end the run and pick the best admitted snapshot")
    submit.add_argument("run_dir")
    submit.set_defaults(func=_cmd_submit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, YamlLoadError) as exc:
        _print_error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
