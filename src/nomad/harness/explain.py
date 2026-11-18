"""``nomad explain CODE`` documentation for the initial diagnostic codes."""

from __future__ import annotations

EXPLAIN: dict[str, str] = {
    "P-SCHEMA": (
        "A YAML document does not match its pydantic schema: missing required key, wrong type, "
        "or an expression that cannot be parsed. Fix the pointed-to line."
    ),
    "P-UNKNOWN-KEY": (
        "A mapping contains a key the schema does not define. PKIR and the plan are closed "
        "vocabularies; remove the key or use the nearest allowed one."
    ),
    "S-PARTITION": (
        "plan.groups must be an ordered partition of graph.ops: every op exactly once, groups "
        "contiguous in graph order, and every group's pkir file present under kernels/."
    ),
    "S-ORDER": ("A group reads a tensor produced by a later group. Reorder groups topologically."),
    "S-PATTERN": (
        "A group's op kinds do not match the fusion pattern named in plan.groups[*].pattern, "
        "or the pattern is not in catalog.yaml."
    ),
    "S-LAYOUT": (
        "A layout names an unknown tensor, a tensor that does not cross a group boundary, or a "
        "layout not in the catalog's layout set."
    ),
    "S-REWRITE": (
        "A rewrite rule is unknown, duplicated, or its side condition does not hold on this "
        "graph (for example fold_norm_gain needs a weight gain feeding a linear)."
    ),
    "S-CAPABILITY": (
        "A PKIR construct is not available on the target DeviceSpec, or a region does not match "
        "its group boundary (kernel name, tensors, out flags, roles, slots)."
    ),
    "S-SLICE": (
        "A slice names an undeclared tensor or buffer, is empty, exceeds its extent without "
        "pad: zero / mask: true, or addresses a staged buffer without a trip index."
    ),
    "S-PLACEMENT": (
        "placement is not keyed by the target device, or warps/warp_tile exceed the device's "
        "runtime or do not divide the tile."
    ),
    "S-CAPACITY": (
        "The region's shared or register footprint exceeds the target DeviceSpec's memory tier. "
        "Reduce tile sizes or stages."
    ),
    "S-UNSUPPORTED": (
        "The group needs a construct PKIR v0.1 cannot express (reductions, online softmax, "
        "layout transforms, or a norm-GEMM epilogue). It is scaffolded but cannot be admitted "
        "until PKIR v0.2."
    ),
    "V-OOB": "Verifier: a slot or operand names a buffer the region does not declare.",
    "V-RACE": (
        "Verifier: a declared copy/matmul/store edge is missing, or a slot is malformed. "
        "Async producers must signal full; staged consumers must wait for free."
    ),
    "V-DEADLOCK": "Verifier: slot cycle (reserved; the v0 slot verifier does not emit it yet).",
    "V-EQUIV": (
        "Verifier: symbolic equivalence is reserved; G4 currently supplies the oracle check."
    ),
    "L-TARGET": "Lowering: no template exists for the target DeviceSpec.",
    "L-TEMPLATE": (
        "Lowering: the region does not match the available template (shape, slice, operand "
        "layout, or placement). The pointer names the rejected field."
    ),
    "L-RESOURCE": (
        "Lowering: register or shared over-subscription; change tiles, stages or placement."
    ),
    "L-TOOLCHAIN": "Lowering: the required compiler (nvcc) is not available on this machine.",
    "L-COMPILE": "Lowering: generated CUDA failed to compile; see log.jsonl for compiler output.",
    "D-UNAVAILABLE": "Difftest: this host has no CUDA runtime, so no lowered kernel can run.",
    "D-MISMATCH": "Difftest: lowered output differs from the PyTorch reference over a shape.",
    "D-NONFINITE": (
        "Difftest found NaN/Inf (reserved; compare uses non-finite values as mismatches)."
    ),
    "D-CRASH": "Difftest: the lowered extension failed to build or the kernel raised.",
}
