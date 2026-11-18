"""G1 plan checks (agent-contract §6): partition, order, pattern, layout, rewrite, support."""

from __future__ import annotations

from pathlib import Path

from nomad.diagnostics import Diagnostic
from nomad.graph.catalog import Catalog
from nomad.graph.plan import Group, PlanDoc
from nomad.graph.rewrites import apply_rule, initial_facts
from nomad.graph.schema import GraphDoc
from nomad.yaml_lines import LineMap, loc_line, loc_pointer


def producer_map(graph: GraphDoc) -> dict[str, str]:
    producers: dict[str, str] = {}
    for op in graph.ops:
        for tensor in op.out_:
            producers[tensor] = op.id
    return producers


def _group_index(plan: PlanDoc) -> dict[str, int]:
    return {op_id: index for index, group in enumerate(plan.groups) for op_id in group.ops}


def group_kinds(graph: GraphDoc, group: Group) -> list[str]:
    return [graph.op(op_id).kind for op_id in group.ops]


def group_inputs(graph: GraphDoc, plan: PlanDoc, group: Group) -> list[str]:
    producers = producer_map(graph)
    inside = set(group.ops)
    inputs: list[str] = []
    for op_id in group.ops:
        for tensor in graph.op(op_id).in_:
            producer = producers.get(tensor)
            if producer is None or producer not in inside:
                if tensor not in inputs:
                    inputs.append(tensor)
    return inputs


def group_outputs(graph: GraphDoc, plan: PlanDoc, group: Group) -> list[str]:
    group_of = {op_id: g.id for g in plan.groups for op_id in g.ops}
    consumers: dict[str, set[str]] = {}
    for op in graph.ops:
        for tensor in op.in_:
            consumers.setdefault(tensor, set()).add(op.id)
    outputs: list[str] = []
    for op_id in group.ops:
        for tensor in graph.op(op_id).out_:
            outside = any(
                group_of.get(consumer) != group.id for consumer in consumers.get(tensor, set())
            )
            if outside or graph.tensors[tensor].out:
                if tensor not in outputs:
                    outputs.append(tensor)
    return outputs


def is_supported_group(graph: GraphDoc, catalog: Catalog, group: Group) -> bool:
    kinds = group_kinds(graph, group)
    pattern = catalog.pattern(group.pattern)
    if not pattern.supported:
        return False
    return pattern.id == "single" and kinds == ["linear"]


def _diagnostic(
    code: str,
    message: str,
    lines: LineMap,
    path: tuple,
    group: str | None = None,
    hint: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=message,
        file="plan.yaml",
        line=loc_line(lines, path),
        pointer=loc_pointer(path),
        group=group,
        hint=hint,
    )


def _check_partition(graph: GraphDoc, plan: PlanDoc, lines: LineMap) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    graph_ops = [op.id for op in graph.ops]
    flat = [op_id for group in plan.groups for op_id in group.ops]
    if flat != graph_ops:
        missing = [op_id for op_id in graph_ops if op_id not in flat]
        extra = [op_id for op_id in flat if op_id not in graph_ops]
        duplicate = sorted({op_id for op_id in flat if flat.count(op_id) > 1})
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unknown {extra}")
        if duplicate:
            detail.append(f"duplicated {duplicate}")
        if not detail:
            detail.append("group ops are out of graph order")
        diags.append(
            _diagnostic(
                "S-PARTITION",
                "groups must be an ordered partition of graph.ops: " + "; ".join(detail),
                lines,
                ("groups",),
                hint="concatenating groups[*].ops must equal graph ops in order",
            )
        )

    seen_groups: set[str] = set()
    seen_pkir: set[str] = set()
    for index, group in enumerate(plan.groups):
        path = ("groups", index)
        if group.id in seen_groups:
            diags.append(
                _diagnostic(
                    "S-PARTITION", f"duplicate group id {group.id!r}", lines, path, group.id
                )
            )
        seen_groups.add(group.id)
        if group.pkir in seen_pkir:
            diags.append(
                _diagnostic(
                    "S-PARTITION", f"duplicate pkir path {group.pkir!r}", lines, path, group.id
                )
            )
        seen_pkir.add(group.pkir)
    return diags


def _check_patterns(
    graph: GraphDoc, plan: PlanDoc, catalog: Catalog, lines: LineMap
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    for index, group in enumerate(plan.groups):
        path = ("groups", index)
        try:
            pattern = catalog.pattern(group.pattern)
        except KeyError:
            diags.append(
                _diagnostic(
                    "S-PATTERN",
                    f"unknown fusion pattern {group.pattern!r}; catalog has "
                    + ", ".join(p.id for p in catalog.patterns),
                    lines,
                    path,
                    group.id,
                )
            )
            continue
        kinds = group_kinds(graph, group)
        if not catalog.matches(group.pattern, kinds):
            expected = "*" if pattern.ops == ["*"] else " -> ".join(pattern.ops)
            diags.append(
                _diagnostic(
                    "S-PATTERN",
                    f"group {group.id!r} has ops {kinds} but pattern "
                    f"{pattern.id!r} expects {expected}",
                    lines,
                    path,
                    group.id,
                )
            )
    return diags


def _check_order(graph: GraphDoc, plan: PlanDoc, lines: LineMap) -> list[Diagnostic]:
    producers = producer_map(graph)
    group_of = _group_index(plan)
    diags: list[Diagnostic] = []
    for index, group in enumerate(plan.groups):
        for op_id in group.ops:
            for tensor in graph.op(op_id).in_:
                producer = producers.get(tensor)
                if producer is None:
                    continue
                producer_group = group_of.get(producer)
                if producer_group is None:
                    continue
                if producer_group > index:
                    diags.append(
                        _diagnostic(
                            "S-ORDER",
                            f"group {group.id!r} reads {tensor!r} produced by later group "
                            f"{plan.groups[producer_group].id!r}",
                            lines,
                            ("groups", index),
                            group.id,
                        )
                    )
                    break
            if diags:
                break
        if diags:
            break
    return diags


def _check_layouts(
    graph: GraphDoc, plan: PlanDoc, catalog: Catalog, lines: LineMap
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    for tensor_name, layout in plan.layouts.items():
        if tensor_name not in graph.tensors:
            diags.append(
                _diagnostic(
                    "S-LAYOUT",
                    f"layout names unknown tensor {tensor_name!r}",
                    lines,
                    ("layouts", tensor_name),
                )
            )
            continue
        allowed = catalog.layouts.get(tensor_name)
        if allowed is None:
            diags.append(
                _diagnostic(
                    "S-LAYOUT",
                    f"tensor {tensor_name!r} has no layout set in the catalog",
                    lines,
                    ("layouts", tensor_name),
                )
            )
            continue
        if layout not in allowed:
            diags.append(
                _diagnostic(
                    "S-LAYOUT",
                    f"layout {layout!r} is not in the catalog for {tensor_name!r}; "
                    f"allowed: {allowed}",
                    lines,
                    ("layouts", tensor_name),
                )
            )
    return diags


def _check_rewrites(
    graph: GraphDoc, plan: PlanDoc, catalog: Catalog, lines: LineMap
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    facts = initial_facts(graph)
    seen: set[str] = set()
    for index, rewrite in enumerate(plan.rewrites):
        path = ("rewrites", index)
        if rewrite.rule not in {rule.id for rule in catalog.rewrites}:
            diags.append(
                _diagnostic(
                    "S-REWRITE",
                    f"unknown rewrite rule {rewrite.rule!r}; catalog has "
                    + ", ".join(rule.id for rule in catalog.rewrites),
                    lines,
                    path,
                )
            )
            continue
        if rewrite.rule in seen:
            diags.append(
                _diagnostic("S-REWRITE", f"duplicate rewrite rule {rewrite.rule!r}", lines, path)
            )
            continue
        seen.add(rewrite.rule)
        facts, reason = apply_rule(facts, rewrite.rule)
        if reason is not None:
            diags.append(
                _diagnostic(
                    "S-REWRITE",
                    f"{rewrite.rule!r} cannot apply here: {reason}",
                    lines,
                    path,
                    hint="rewrites are order-sensitive; check the earlier rules in plan.rewrites",
                )
            )
            break
    return diags


def _check_supported(
    graph: GraphDoc, plan: PlanDoc, catalog: Catalog, lines: LineMap
) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    for index, group in enumerate(plan.groups):
        if is_supported_group(graph, catalog, group):
            continue
        kinds = group_kinds(graph, group)
        diags.append(
            _diagnostic(
                "S-UNSUPPORTED",
                f"group {group.id!r} uses pattern {group.pattern!r} over {kinds}; "
                "PKIR v0.1 can lower only single linear regions",
                lines,
                ("groups", index),
                group.id,
                hint="reductions, online softmax and layout transforms are in pkir-spec §9",
            )
        )
    return diags


def check_plan(
    plan: PlanDoc,
    graph: GraphDoc,
    catalog: Catalog,
    run_dir: str | Path,
    lines: LineMap,
) -> list[Diagnostic]:
    """Run G1 plan checks. Returns diagnostics in gate order, empty when the plan is legal."""

    run_dir = Path(run_dir)
    diags: list[Diagnostic] = []
    if plan.graph != graph.graph:
        diags.append(
            _diagnostic(
                "S-PARTITION",
                f"plan targets graph {plan.graph!r}, but graph.yaml is {graph.graph!r}",
                lines,
                ("graph",),
            )
        )
    if diags:
        return diags
    diags = _check_partition(graph, plan, lines)
    if not diags:
        diags = _check_patterns(graph, plan, catalog, lines)
    if not diags:
        diags = _check_order(graph, plan, lines)
    if not diags:
        for index, group in enumerate(plan.groups):
            path = Path(group.pkir)
            if path.is_absolute() or ".." in path.parts or not str(path).endswith(".pkir.yaml"):
                diags.append(
                    _diagnostic(
                        "S-PARTITION",
                        f"pkir path {group.pkir!r} must be a relative kernels/*.pkir.yaml path",
                        lines,
                        ("groups", index),
                        group.id,
                    )
                )
                continue
            if not (run_dir / path).is_file():
                diags.append(
                    _diagnostic(
                        "S-PARTITION",
                        f"group {group.id!r} names missing PKIR file {group.pkir!r}",
                        lines,
                        ("groups", index),
                        group.id,
                        hint="run `nomad scaffold " + group.id + "`",
                    )
                )
    if not diags:
        diags = _check_layouts(graph, plan, catalog, lines)
    if not diags:
        diags = _check_rewrites(graph, plan, catalog, lines)
    if not diags:
        diags = _check_supported(graph, plan, catalog, lines)
    return diags
