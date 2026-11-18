"""Rewrite exploration is over subsets (which rules) and orders (when they apply)."""

from __future__ import annotations

from nomad.graph.catalog import load_catalog
from nomad.graph.rewrites import apply_rule, feasible_sequences, initial_facts
from nomad.graph.schema import GraphDoc
from nomad.reference.block import graph_attention_block


def block():
    return GraphDoc.model_validate(graph_attention_block())


def test_order_matters_for_norm_rules():
    graph = block()
    facts = initial_facts(graph)
    assert facts.norm_to_linear and facts.norm_has_gain
    assert apply_rule(facts, "norm_as_row_scale")[0] is None
    after_fold, reason = apply_rule(facts, "fold_norm_gain")
    assert reason is None and after_fold is not None
    assert apply_rule(after_fold, "norm_as_row_scale")[0] is not None


def test_feasible_sequences_are_order_sensitive():
    sequences = feasible_sequences(block(), load_catalog())
    assert ("fold_norm_gain", "norm_as_row_scale") in sequences
    assert ("norm_as_row_scale", "fold_norm_gain") not in sequences
    assert ("norm_as_row_scale",) not in sequences
    assert len(sequences) == 8
    assert all(len(set(sequence)) == len(sequence) for sequence in sequences)


def test_depth_limit():
    sequences = feasible_sequences(block(), load_catalog(), max_depth=1)
    assert {sequence[0] for sequence in sequences} == {"fold_norm_gain", "fold_attn_scale"}
    assert all(len(sequence) == 1 for sequence in sequences)
