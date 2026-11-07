# Agent Contract v0.1 — what the agent reads, writes, calls and gets back

v0.1 draft (agent driver; to be agreed by HK and OS before I2)

## §1 Purpose and scope

The agent is an off-the-shelf coding agent. NOMAD does not train or modify it; this contract is its
entire interface to the system. It fixes four things: the files the agent may read and edit (§2–§4),
the commands it may run (§5), the gates every submission passes through (§6), and the diagnostics it
gets back (§7). §8 fixes when a run ends and what counts as its result; §9 maps run configurations to
the proposal's experiments.

The agent makes the device-independent decisions — fusion boundaries, layouts, sync edges, staging —
and nothing else. It never writes CUDA or NKI, never runs a compiler, and never sees device-specific
text: lowering is template code under `src/nomad/lowering/`, and diagnostics are phrased in plan and
PKIR terms (§7). That is what lets the same agent and the same diagnostics serve `sm86` and `trn2`.

## §2 Run workspace

`nomad init` (harness-only, §5) creates one directory per run under `results/runs/<run-id>/`:

|path|written by|agent may|contents|
|---|---|---|---|
|`task.yaml`|harness|read|run configuration (§9)|
|`devicespec.yaml`|harness|read|copy of the target `DeviceSpec`|
|`graph.yaml`|harness|read|the block's op graph (§3)|
|`catalog.yaml`|harness|read|fusion catalog, layout set, rewrite rules (§4)|
|`plan.yaml`|agent|edit (graph mode only)|fusion plan, layouts, rewrites (§4)|
|`kernels/<group>.pkir.yaml`|agent|edit|one PKIR region per fused group (`docs/pkir-spec.md`)|
|`diagnostics/latest.json`|harness|read|report of the last `nomad check` (§7)|
|`snapshots/<n>/`|harness|—|copy of every admitted submission|
|`log.jsonl`|harness|—|one line per command: inputs hash, gate outcomes, metrics, tokens|

`nomad check` hashes every read-only file and refuses to run (exit 2) if one changed. In kernel mode
(§9) `plan.yaml` is read-only too.

## §3 Block graph (input)

`graph.yaml` is generated from `nomad.reference.block` and names every op the plan partitions. Shapes
are symbolic; the shape table (`task.shapes`) binds them.

```yaml
graph: attention_block
params: [B, S, D, H]
assume: ["D % H == 0"]
tensors:
  x:   {shape: [B * S, D], dtype: f16}
  g:   {shape: [D], dtype: f16, weight: true}
  w_qkv: {shape: [3 * D, D], dtype: f16, weight: true}
  w_o: {shape: [D, D], dtype: f16, weight: true}
  h:   {shape: [B * S, D], dtype: f16}
  qkv: {shape: [B * S, 3 * D], dtype: f16}
  q:   {shape: [B, H, S, D / H], dtype: f16}
  k:   {shape: [B, H, S, D / H], dtype: f16}
  v:   {shape: [B, H, S, D / H], dtype: f16}
  o:   {shape: [B, H, S, D / H], dtype: f16}
  o2:  {shape: [B * S, D], dtype: f16}
  y:   {shape: [B * S, D], dtype: f16, out: true}
ops:
  - {id: norm,  kind: rmsnorm, in: [x, g], out: [h], eps: 1.0e-6}
  - {id: qkv,   kind: linear,  in: [h, w_qkv], out: [qkv]}
  - {id: split, kind: layout,  in: [qkv], out: [q, k, v]}
  - {id: attn,  kind: sdpa,    in: [q, k, v], out: [o], causal: false}
  - {id: merge, kind: layout,  in: [o], out: [o2]}
  - {id: out,   kind: linear,  in: [o2, w_o], out: [y]}
```

Every op kind has a PyTorch reference in `nomad.reference`; a group's reference is the composition of
its ops' references. That per-group reference is what makes difftest and equivalence failures name a
group instead of the whole block (§6).

## §4 Plan document (`plan.yaml`) and catalog

```yaml
plan: 0
graph: attention_block
groups:
  - {id: norm_qkv, pattern: norm_gemm, ops: [norm, qkv], pkir: kernels/norm_qkv.pkir.yaml}
  - {id: attention, pattern: attention, ops: [split, attn, merge], pkir: kernels/attention.pkir.yaml}
  - {id: out_proj, pattern: single, ops: [out], pkir: kernels/out_proj.pkir.yaml}
layouts:
  qkv: "t b h s d"
  o: "b s h d"
rewrites:
  - {rule: fold_norm_gain}
  - {rule: norm_as_row_scale}
```

Field rules, all checked by the static gate (§6):

- `groups` is the fusion plan: an ordered partition of `graph.ops`. Every op is in exactly one group,
  groups are in a topological order of the op graph, and a group's op sequence matches its `pattern`
  in the fusion catalog. Each group names the PKIR file that realizes it.
- `layouts` assigns a layout from the layout set to tensors that cross a group boundary, in einops
  notation over the graph's dims (`t` indexes q/k/v). A tensor not listed keeps its reference layout.
  When a boundary layout already matches what a consumer needs, the corresponding `layout` op becomes
  an index map inside the producer's or consumer's region rather than a data movement.
- `rewrites` lists catalog rules applied to the graph before grouping. A rewrite is admitted only if
  its side condition holds; its numeric effect is then judged by difftest like any other change.

Catalog v0 (`catalog.yaml`; entries are added by PR, never by the agent):

|fusion pattern|ops|note|
|---|---|---|
|`single`|any one op|always legal; the unfused start state (§8)|
|`norm_gemm`|`rmsnorm` → `linear`|row statistics as a reduction, applied as a GEMM epilogue row scale; needs `norm_as_row_scale`|
|`attention`|`layout` → `sdpa` → `layout`|head split and merge folded into the attention region's index maps|
|`attn_out`|`layout` → `sdpa` → `layout` → `linear`|attention output feeds out-proj without a round trip to memory (stretch)|

|rewrite rule|from → to|side condition|
|---|---|---|
|`fold_norm_gain`|`linear(rmsnorm(x; g), W)` → `linear(rmsnorm(x; 1), W · diag(g))`|`g`, `W` are weights|
|`norm_as_row_scale`|`linear(rmsnorm(x; 1), W)` → `rowscale(linear(x, W), rsqrt(mean(x²) + eps))`|none|
|`fold_attn_scale`|scale `q` by `1/sqrt(D/H)` → scale the `q` rows of `w_qkv` once|`w_qkv` is a weight|

Weight transforms implied by rewrites run once, at load, and are not timed — the same treatment
`torch.compile` gets for constant folding. This must be stated in the report next to the latency table.

|layout set v0|tensor|
|---|---|
|`"(b s) (t h d)"` (reference), `"t b h s d"`|`qkv`|
|`"b h s d"` (reference), `"b s h d"` (makes `merge` a free view)|`o`|

PKIR v0.1 can express only the `linear` regions. `rmsnorm`, `sdpa` and the `norm_gemm` epilogue need
the reductions, transcendentals and running state listed in `docs/pkir-spec.md` §9; until PKIR v0.2
lands, groups containing them are scaffolded but fail the static gate with `S-UNSUPPORTED`.

## §5 Commands

The agent's shell is restricted to file reads, edits inside its workspace, and these commands:

|command|cost|what it does|
|---|---|---|
|`nomad check [--upto GATE]`|1 submission|runs the gates of §6 in order, stops at the first failure, writes `diagnostics/latest.json`, prints its summary|
|`nomad cost [GROUP]`|free|cost-model estimate (§6, G5) for the current files, without lowering or running|
|`nomad scaffold GROUP`|free|(re)writes `kernels/GROUP.pkir.yaml` from the template of the group's catalog pattern|
|`nomad explain CODE`|free|prints the documentation for a diagnostic code (§7)|
|`nomad submit`|free|ends the run (§8)|

`nomad init TASK RUN_DIR` is harness-only. Exit codes: `0` admitted, `1` rejected at a gate, `2`
harness error (not charged to the budget, logged for us to fix), `3` budget exhausted. Compilers,
Python and the GPU are not reachable from the agent's shell, so no gate can be bypassed.

## §6 Gates

Every `nomad check` runs these in order and stops at the first gate that fails. A gate never repairs
a submission.

|gate|checks|owner|failure codes|
|---|---|---|---|
|G0 parse|YAML form and schema of `plan.yaml` and every PKIR file (pydantic models; unknown keys are errors)|OS|`P-*`|
|G1 static|plan legality (§4 rules); PKIR types and capabilities against `devicespec.yaml`; slice safety and placement obligations (`pkir-spec` §5); shared and register footprint against tier capacity (`pkir-spec` §6)|OS (PKIR), HK (plan)|`S-*`|
|G2 verify|memory safety, race and deadlock freedom from slot edges, and equivalence of each region to its group's reference|OS|`V-*`|
|G3 lower|template lowering of every region for the target device, then compile (nvcc on `sm86`, NEFF compile on `trn2`)|HK (CUDA), OS (NKI)|`L-*`|
|G4 difftest|each group against its reference, then the whole block against `reference_fp32`, over the shape table at atol = rtol = 2e-2|HK|`D-*`|
|G5 time|`sm86`: median of 100 timed calls after 10 warmups per shape (CUDA events); `trn2`: cost-model estimate, marked `estimated`|HK|none|

Rules:

- The cost model never rejects. It is feedback (`nomad cost`, and G5 on devices without hardware in
  the loop). Capacity overflow is a G1 rejection because it is a static fact, not an estimate.
- A G3 failure after G1 and G2 passed is a lowering bug by definition (`pkir-spec` §8) and returns
  exit `2`, except `L-RESOURCE` (register or shared over-subscription), which the agent can fix by
  changing tiles or placement.
- G5 reports, per shape, the time and the speedup over the fastest of eager, SDPA and `torch.compile`,
  and their geometric mean across shapes. Timing is not a gate: a slow admitted kernel is admitted.

## §7 Diagnostics

`diagnostics/latest.json`, one per `nomad check`:

```json
{
  "contract": 0,
  "submission": 7,
  "verdict": "rejected",
  "gate": "difftest",
  "diagnostics": [
    {
      "code": "D-MISMATCH",
      "file": "kernels/norm_qkv.pkir.yaml",
      "line": 24,
      "pointer": "/body",
      "group": "norm_qkv",
      "message": "qkv differs from reference on shape b2s77d1024h16: 3072 of 473088 elements, worst |err| 1.9 at qkv[153, 2047]",
      "where": {"tensor": "qkv", "index": [153, 2047], "tile": {"bm": 2, "bn": 31}, "kt": null},
      "hint": "row 153 is in the ragged M tile bm = 2 (valid rows 128..153); check pad: zero on the X copy"
    }
  ],
  "metrics": null,
  "budget": {"checks_left": 33, "tokens_used": 412580}
}
```

Rules:

- Device-free: messages name plan paths, PKIR lines and steps, and `DeviceSpec` keys
  (`capabilities.tensor_core_mma`), never vendor instructions or compiler output. Raw tool output goes
  to `log.jsonl` only.
- Localized: every diagnostic names a file and line; PKIR steps are addressed by JSON Pointer
  (RFC 6901) into the document. Difftest mismatches are mapped back to the output tile and K-trip
  that produced the worst element.
- Bounded: only the failing gate's diagnostics, at most 5, each `message` at most 300 characters, so
  a report costs the agent a predictable number of tokens.
- `hint` is optional and never names a fix that changes semantics.
- On `verdict: admitted`, `diagnostics` is empty and `metrics` carries the G5 numbers.

Code prefixes match the gates: `P-` parse, `S-` static, `V-` verify, `L-` lower, `D-` difftest.
Initial codes: `P-SCHEMA`, `P-UNKNOWN-KEY`, `S-PARTITION`, `S-ORDER`, `S-PATTERN`, `S-LAYOUT`,
`S-REWRITE`, `S-CAPABILITY`, `S-SLICE`, `S-PLACEMENT`, `S-CAPACITY`, `S-UNSUPPORTED`, `V-OOB`,
`V-RACE`, `V-DEADLOCK`, `V-EQUIV`, `L-RESOURCE`, `D-MISMATCH`, `D-NONFINITE`, `D-CRASH`. Codes are
stable once released; `nomad explain` documents each one.

## §8 Start state, stop rule, result

- **Start.** `start: unfused` (the default) writes a plan with one `single` group per op, no layouts
  and no rewrites, and scaffolds every region from its template. This start is always valid, gives a
  measured baseline for the agent's improvement, and does not leak our hand-written schedules into the
  run. `start: <path>` begins from a given plan instead; kernel mode always uses it.
- **Stop.** The run ends at the first of: the agent runs `nomad submit`; `budget.checks` submissions
  used; `budget.tokens` agent tokens used (input + output, as reported by the agent's API and summed by
  the driver); `budget.wall_clock_min` elapsed.
- **Result.** The admitted snapshot with the best geomean time, not the last state. A run with no
  admitted snapshot is a failed run and is reported as such, never dropped.

## §9 Run configuration (`task.yaml`) and the experiments

```yaml
contract: 0
task: attention_block
device: sm86
shapes: block_v0
mode: graph
gates: full
start: unfused
seed: 0
budget: {checks: 40, tokens: 2000000, wall_clock_min: 120}
```

|knob|values|experiment (proposal §3)|
|---|---|---|
|`mode`|`graph` (edits plan and kernels) / `kernel` (plan fixed by `start`, edits kernels only)|(4) kernel-only vs graph-level agent|
|`gates`|`full` / `tests-only` (G0, G3, G4, G5: what a KernelBench-style loop sees)|(2) agent with vs without gates|
|`device`|`sm86` / `trn2`|(3) portability: rerun an admitted `sm86` workspace with only `devicespec.yaml` swapped; retarget cost = checks and tokens until admitted on `trn2`|
|—|admitted results at `gates: full`|(1) block latency vs eager, `torch.compile`, SDPA|

The mutant half of (2) does not use the agent: each mutant is an admitted workspace with one injected
bug (`mutants/`), run through `nomad check` in a fresh process per mutant, recording which gate first
rejects it.

## §10 Open questions

- Hand-off format for the agent's system prompt: which of `pkir-spec.md`, this contract and the
  catalog are inlined, and which it reads on demand.
- Whether `hint` is on in every configuration or ablated.
- Budget numbers: the values above are placeholders until the pilot runs measure tokens per check.
- Whether `attn_out` is in scope for the report or stays a stretch pattern.