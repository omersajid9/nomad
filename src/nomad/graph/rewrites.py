"""Rewrite exploration: which catalog rules apply, in which order (agent-contract §4).

The two axes are the rule subset and its order. Feasibility is sequential: a rule may only
apply after its preconditions hold in the state produced by the previous rules. The fact state
below is enough for the v0 catalog; a structural rewriter replaces it when PKIR v0.2 lands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from nomad.graph.catalog import Catalog
from nomad.graph.schema import GraphDoc


@dataclass(frozen=True)
class Facts:
    norm_to_linear: bool
    norm_has_gain: bool
    weighted_linear_feeds_sdpa: bool
    attn_scale_folded: bool = False


def _producer_map(graph: GraphDoc) -> dict[str, str]:
    producers: dict[str, str] = {}
    for op in graph.ops:
        for tensor in op.out_:
            producers[tensor] = op.id
    return producers


def _norm_to_linear(graph: GraphDoc) -> bool:
    producers = _producer_map(graph)
    return any(
        op.kind == "linear"
        and any(
            (producer := producers.get(tensor)) and graph.op(producer).kind == "rmsnorm"
            for tensor in op.in_
        )
        for op in graph.ops
    )


def _norm_has_gain(graph: GraphDoc) -> bool:
    return any(
        op.kind == "rmsnorm" and any(graph.tensors[t].weight for t in op.in_) for op in graph.ops
    )


def _linear_feeds_sdpa(graph: GraphDoc) -> bool:
    producers = _producer_map(graph)
    for op in graph.ops:
        if op.kind != "sdpa":
            continue
        stack, seen = list(op.in_), set()
        while stack:
            producer = producers.get(stack.pop())
            if producer is None or producer in seen:
                continue
            seen.add(producer)
            producer_op = graph.op(producer)
            if producer_op.kind == "linear" and any(
                graph.tensors[t].weight for t in producer_op.in_
            ):
                return True
            stack.extend(producer_op.in_)
    return False


def initial_facts(graph: GraphDoc) -> Facts:
    return Facts(
        norm_to_linear=_norm_to_linear(graph),
        norm_has_gain=_norm_has_gain(graph),
        weighted_linear_feeds_sdpa=_linear_feeds_sdpa(graph),
    )


def apply_rule(facts: Facts, rule: str) -> tuple[Facts | None, str | None]:
    """Return ``(new_facts, None)`` or ``(None, reason)`` when the precondition fails."""

    if rule == "fold_norm_gain":
        if not facts.norm_to_linear:
            return None, "requires an rmsnorm feeding a linear"
        if not facts.norm_has_gain:
            return None, "requires the norm gain to still be present"
        return replace(facts, norm_has_gain=False), None
    if rule == "norm_as_row_scale":
        if not facts.norm_to_linear:
            return None, "requires an rmsnorm feeding a linear"
        if facts.norm_has_gain:
            return None, "requires the gain folded first: apply fold_norm_gain before this rule"
        return replace(facts, norm_to_linear=False), None
    if rule == "fold_attn_scale":
        if not facts.weighted_linear_feeds_sdpa:
            return None, "requires a weighted linear feeding an sdpa"
        return replace(facts, attn_scale_folded=True), None
    return None, f"unknown rewrite rule {rule!r}"


def feasible_sequences(
    graph: GraphDoc, catalog: Catalog, *, max_depth: int | None = None
) -> list[tuple[str, ...]]:
    """Every feasible ordered sequence of distinct catalog rules.

    Order matters: a sequence is kept only when each rule's precondition holds after the
    previous rules. The same sequence reached by different bindings collapses here; binding
    (which match site) is the third axis and lands with the structural rewriter.
    """

    rules = [rule.id for rule in catalog.rewrites]
    depth = len(rules) if max_depth is None else min(max_depth, len(rules))
    found: list[tuple[str, ...]] = []

    def walk(facts: Facts, sequence: tuple[str, ...]) -> None:
        if sequence:
            found.append(sequence)
        if len(sequence) >= depth:
            return
        for rule in rules:
            if rule in sequence:
                continue
            new_facts, _ = apply_rule(facts, rule)
            if new_facts is not None:
                walk(new_facts, sequence + (rule,))

    walk(initial_facts(graph), ())
    return sorted(found, key=lambda seq: (len(seq), seq))
