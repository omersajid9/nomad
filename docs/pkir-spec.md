# PKIR v0.1 — Portable Kernel IR Specification

v0.1 (integration point I1)

## §1 Purpose and scope

PKIR is a tile-level dataflow IR over memory-tier buffers, worker roles, staged pipelines, slot-based
synchronization and affine slices, parameterized by a `DeviceSpec`. Device instructions appear only
as capability-gated ops carrying semantic contracts (§4): the body of a PKIR program is device-free,
and every device-specific number lives in `placement`.

PKIR is not a CUDA or PTX dialect. It has no thread indices, no registers-as-variables, no inline
assembly and no vendor intrinsics. It is also not the graph-level IR: fusion plans, layout sets and
gated rewrites live above it in the agent action space (DNNFusion-, Einsum-Tree- and TASO-style) and
compile down into regions of the form specified here. The design decisions recorded below are
traceable to the literature annotated in [`docs/refs/index.md`](refs/index.md), which lists the spec
section each paper motivates.

Scope of v0.1: exactly the constructs exercised by the QKV projection region of the fused attention
block — tile GEMM, staging, a two-stage pipeline, roles, slot edges and affine slices. §9 states the
widening contract for everything else the block needs (RMSNorm, SDPA with online softmax, out-proj).

## §2 Document form and tooling

A PKIR program is a single-document YAML mapping validated by the pydantic models at
`src/nomad/pkir/models.py`, from which `schemas/pkir.schema.json` is emitted for the agent contracts.
Slices and trip counts are affine strings parsed into ASTs by `src/nomad/pkir/expr.py`; that
module's canonical printer is the reference for round-tripping, so a printed document validates and
re-parses to the same AST. YAML is already a dependency (`pyyaml`), and the models are the single
source of truth for the schema that the agent sees.

Normative conventions:

- one YAML document per program; aliases, anchors, tags and multi-document streams are errors;
- keys are `snake_case`; role, buffer and tensor names match `[a-z][a-z0-9_]*`;
- `pkir` is an integer revision, `0` for this draft;
- flow style for leaf mappings and block style for lists, as in the §7 listing, so documents follow
  one formatting;
- one space around binary operators, and parentheses only where the reading is otherwise ambiguous,
  so that semantically equal documents are byte-equal and listings are diff-stable;
- unknown keys are schema errors, never warnings;
- every parse or check failure reports the YAML line it came from.

## §3 Core objects

Top-level keys, in document order:

|key|type|meaning|
|---|---|---|
|`pkir`|int|revision; `0` for this draft|
|`kernel`|str|entry symbol the lowering emits|
|`params`|list of names|symbolic tensor dims|
|`assume`|list of predicates|preconditions the region requires of its caller|
|`tensors`|map|name → shape/dtype/out|
|`tile`|map|the region's tile sizes|
|`grid`|map|block-index extents|
|`roles`|map|name → kind|
|`buffers`|map|name → tier-resident tile buffer|
|`placement`|map|device name → placement block|
|`body`|list of steps|the region's schedule|

Field semantics:

- `pkir`: integer revision, `0`.
- `kernel`: the entry symbol the lowering emits (`qkv_proj`).
- `params`: list of symbolic tensor dims (`[M, N, K]`); symbols denote non-negative integers unless
  an `assume` predicate says otherwise.
- `assume`: list of affine predicates over `params` that the region requires of its caller; the
  worked example states `["K % 8 == 0"]`, the region precondition at `tests/gpu/qkv_tile_kernel.cu:151`.
  A caller violating `assume` is a host-side error, not undefined behavior.
- `tensors`: name → `{shape: [expr, ...], dtype: f16|f32, out: true?}`. Row-major strides are implied
  by `shape`; v0.1 has no strided-tensor construct. Example: `X [M, K] f16`, `W [N, K] f16`,
  `Y [M, N] f16` with `out: true`.
- `tile`: `{m: 64, n: 64, k: 32}` — the only place tile sizes are written; every other expression
  references `tile.m`, `tile.n` or `tile.k`.
- `grid`: `{bm: "ceil(M / tile.m)", bn: "ceil(N / tile.n)"}` — block-index extents along M and N.
  `bm` and `bn` are in scope inside `body`; `grid.bm × grid.bn` is the launch's block count.
- `roles`: name → `{kind: move|matrix}`. `move` copies between tiers; `matrix` runs tensor-core work.
  `reduce` and `elementwise` are reserved names not specified in v0.1. Role membership is a placement
  decision: with `specialize: false` every warp executes every role's steps in program order.
- `buffers`: name → `{tier: shared|register, shape: [expr, expr], dtype, stages?: int, stride?: [expr, expr]}`.
  `stages` present ⇒ a staged pipeline slot array addressed `buf[trip]`; `stride` overrides the
  implied row stride and exists to describe padding the lowering must preserve (`A`/`B`:
  `[tile.k + 8, 1]`, `C`: `[tile.n + 4, 1]`).
- `placement`: device name → `{warps: int, specialize: bool, warp_tile: {mma: [expr, expr]}}`. v0.1
  specifies `sm86` only. Absent fields fall back to `DeviceSpec` defaults, so the same body lowers
  unchanged for a device whose placement block is empty.
- `body`: ordered list of steps, each with an optional `role`.

Step kinds:

|step|fields|
|---|---|
|`fill`|`{role, dst, value}`|
|`for`|`{var, trips, body}` (v0.1: one loop per region)|
|`copy`|`{role, src: slice\|buffer, dst: slice\|buffer, pad?: zero, async?: true, cast?: f16, mask?: true, wait?: [slot], signal?: [slot]}`|
|`matmul`|`{role, acc, a, b, transpose_b?: true, wait?: [slot], signal?: [slot]}`|

- `copy` moves a rectangular slice between tiers. `pad: zero` makes out-of-extent source elements
  read as zero; `mask: true` predicates writes on the destination extent; `async: true` selects the
  device's asynchronous copy engine, whose completion is observed through the waiting slot (§4);
  `cast` converts in flight, rounding once to nearest-even.
- `matmul` accumulates `acc += a · bᵀ` into a `register`-tier buffer; with `transpose_b: true`, `b`
  is laid out `[n, k]` (an `nn.Linear` weight).
- A staged buffer is addressed without a trip index — `A[kt]` names stage `kt mod stages`.
- Slice grammar: `name[r0 : r1, c0 : c1]` with half-open affine ranges, or `name[i]` for a staged
  buffer. Slices, trip counts and `wait`/`signal` slots are the only places expressions appear.
- Expression and name rules: integer literals; `params`; `tile.*`; grid ids and in-scope loop
  variables (`bm`, `bn`, `kt`, and the loop variable of any enclosing `for`); `+ - * / %`,
  `ceil_div(a, b)`, `min`, `max`, comparisons, `&&`, `||`, `!`, parentheses. `/` and `%` are floor
  division on non-negative operands, and every operand is declared non-negative. An undeclared name,
  or a bound that is not affine in declared names, is a schema error reporting its line.

## §4 Capability ops and semantic contracts

Each capability is realized per device; the contract column is what the verifier and the cost model
may assume about the op regardless of realization. The `trn2` column is `pending — NKI lowering
task`; no trn2 op is claimed in v0.1.

|capability (IR form)|contract|sm86 realization|trn2|
|---|---|---|---|
|tile copy (`copy`)|rectangular slice moved between tiers; out-of-extent elements read as zero under `pad: zero`; writes predicated on extent under `mask: true`; `async: true` selects an asynchronous engine whose completion is observed through the waiting slot; `cast` rounds once, round-to-nearest-even|`cp.async.cg.shared.global` in 8-half (16 B) chunks per thread, `cp.async.commit_group` / `cp.async.wait_group N`; chunk width from DeviceSpec `copy_engines.chunk_bytes`|pending — NKI lowering task|
|slot wait/signal (`wait`/`signal`)|per stage `buf[i]`: `signal: full` is a release and `wait: full` an acquire, symmetrically for `free`; `free` holds initially; this state machine is what makes "producer published" equal "consumer may read"|block barrier (`__syncthreads()`) plus, for async producers, the per-thread group wait — the rule behind `qkv_tile_kernel.cu:105-109`|pending — NKI lowering task|
|tensor-core matmul (`matmul`)|`acc += a · bᵀ`, with `transpose_b` selecting the operand layout; fp16 operands, fp32 accumulate; decomposed by `placement.warp_tile.mma` and the device mma shape, so `warp_tile.mma` must be divisible by that shape and operand leading dimensions must satisfy the device rule|`wmma::load_matrix_sync` + `wmma::mma_sync` at `[16,16,16]`, A `row_major` with `ldm = stride`, B `col_major` with `ldm = stride` (a `[N,K]` weight under `transpose_b: true`)|pending — NKI lowering task|
|register→shared move (`copy` from `register` to `shared`)|warp-fragment store, row-major, `ldm` a multiple of 4 floats|`wmma::store_matrix_sync` with `ldm = tile.n + 4`|pending — NKI lowering task|

## §5 Verifier-facing properties

Per construct, the obligation and the gate that consumes it:

|property|obligation|consumed by|
|---|---|---|
|slice safety|every `copy` slice is the intersection of its affine range with the tensor or buffer extent; an unbounded slice is a **slice defect** reported with the offending expression, so the agent fixes the range instead of the checker. `pad: zero` on reads and `mask: true` on the store are what make ragged `M`/`N` safe.|static checks|
|slot discipline|every read waits `full`, every write waits `free`; each stage of a `stages: N` buffer is produced and consumed exactly once per trip; every participant of a role reaches each wait and signal (no divergent control flow across a barrier). Otherwise the region is reported as a race or a deadlock naming the step id.|verifier (race/deadlock freedom)|
|placement obligations|`warp_tile.mma` divides the buffer tile; mma operand leading dimensions satisfy the device rule; the placement's thread map partitions each tile move.|static checks, and for the thread-map partition the differential test and the mutant study|
|equivalence|a `matmul` region equals its reference: zero padding contributes nothing, the store is bounds-checked, accumulation is fp32, and the output is rounded to f16 exactly once.|differential tests|

## §6 Cost-model hooks

The cost model reads only the region and the `DeviceSpec`, never lowering code. Per construct:

|construct|contribution|
|---|---|
|`copy` (global → shared)|bytes moved per stage = product of the slice extents × `element_bytes`; padded columns cost shared capacity but no global traffic; a `pad: zero` chunk that is fully out of extent costs nothing|
|`copy` (shared → global)|bytes written = clipped slice extent × destination element size|
|`copy` (`register` → `shared`)|bytes written = tile area × accumulator size|
|`matmul`|FLOPs per trip = `2 · tile.m · tile.n · tile.k`; total = `2 · M · N · K`|
|`buffers`|shared footprint = Σ over staged buffers of `stages × rows × stride[0] × element_bytes` (padding included); register footprint = area of each `register` buffer|
|`placement`|derived occupancy hint = min(memory-tier capacity / footprint, warp slots / warps)|

Worked numbers for the §7 region on sm86: `A` and `B` are `64 × 40` halves = 5,120 B per stage,
10,240 B per buffer at `stages: 2`; `C` is `64 × 68` floats = 17,408 B; total 37,888 B (≈37 KB) of
static shared per block, which admits at most 2 blocks/SM. `acc` is a 64 × 64 fp32 tile = 4,096
floats, spread by the sm86 placement over 4 warps as 32 × 32 each — the largest single register
consumer, and the next occupancy bound after shared memory. Per block and K-tile the global traffic
is `(tile.m + tile.n) · tile.k · 2 B = 8,192 B`; the epilogue writes `64 × 64 × 2 B = 8,192 B` once.

## §7 Worked example: the QKV projection region

The I1 document. It is the normative example of the v0.1 form: the region is the QKV projection GEMM
`Y[M,N] = X[M,K]·W[N,K]ᵀ` (fp16 in, fp32 accumulate, fp16 out) — the QKV stage of the fused attention
block — whose CUDA realization is the oracle at `tests/gpu/qkv_tile_kernel.cu`. Comments name the
kernel lines each fragment comes from.

```yaml
# PKIR v0.1 — QKV projection region (integration point I1).
# Target oracle: tests/gpu/qkv_tile_kernel.cu (Y[M,N] = X[M,K] * W[N,K]^T, fp16 in, fp32 accumulate).
pkir: 0
kernel: qkv_proj
params: [M, N, K]
assume: ["K % 8 == 0"]                                  # L151: TORCH_CHECK(K > 0 && K % 8 == 0)
tensors:
  X: {shape: [M, K], dtype: f16}                        # L144-150: x[M,K], contiguous, 16-byte aligned
  W: {shape: [N, K], dtype: f16}                        # L144-150: w[N,K], nn.Linear weight layout
  Y: {shape: [M, N], dtype: f16, out: true}
tile: {m: 64, n: 64, k: 32}                             # L30: BM, BN, BK
grid: {bm: "ceil(M / tile.m)", bn: "ceil(N / tile.n)"}  # L158-159: grid.y, grid.x
roles:
  load: {kind: move}                                    # L9-12, L57-75: loader and storer
  mma: {kind: matrix}                                   # L10, L111-124: 4 warps, 2x2 WMMA tiles each
buffers:
  A: {tier: shared, shape: [tile.m, tile.k], dtype: f16, stages: 2, stride: [tile.k + 8, 1]}  # L80, L36: As[STAGES][BM][LDS], LDS = BK + 8
  B: {tier: shared, shape: [tile.n, tile.k], dtype: f16, stages: 2, stride: [tile.k + 8, 1]}  # L81, L36: Bs[STAGES][BN][LDS]
  acc: {tier: register, shape: [tile.m, tile.n], dtype: f32}                                   # L90: WMMA accumulator fragments
  C: {tier: shared, shape: [tile.m, tile.n], dtype: f32, stride: [tile.n + 4, 1]}              # L82, L37: Cs[BM][LDC], LDC = BN + 4
placement:
  sm86: {warps: 4, specialize: false, warp_tile: {mma: [tile.m / 2, tile.n / 2]}}  # L31-33: WARPS_M/N = 2x2, WM/WN = 32x32
body:
  - fill: {role: mma, dst: acc, value: 0.0}             # L90-94: fill_fragment(acc, 0)
  - for:                                                # L99-126: K-tile pipeline + MMA loop
      var: kt
      trips: "ceil(K / tile.k)"                         # L88: k_tiles
      body:
        - copy:                                         # L61-67: load_stage, X half
            role: load
            src: "X[bm*tile.m : bm*tile.m + tile.m, kt*tile.k : kt*tile.k + tile.k]"
            dst: "A[kt]"
            pad: zero                                   # L65: the ok predicate; L46: src-size 0 zero-fills
            async: true                                 # L44-47, L66: 16-byte cp.async.cg chunks
            wait: [A.free]
            signal: [A.full]
        - copy:                                         # L68-74: load_stage, W half
            role: load
            src: "W[bn*tile.n : bn*tile.n + tile.n, kt*tile.k : kt*tile.k + tile.k]"
            dst: "B[kt]"
            pad: zero                                   # L72: the ok predicate
            async: true
            wait: [B.free]
            signal: [B.full]
        - matmul:                                       # L111-124: warp tile + kk loop
            role: mma
            acc: acc
            a: "A[kt]"
            b: "B[kt]"
            transpose_b: true                           # L115: col_major B over a [N,K] weight
            wait: [A.full, B.full]                      # L105-109: cp.async.wait_group + __syncthreads
            signal: [A.free, B.free]                    # L125: __syncthreads
  - copy:                                               # L130-134: store_matrix_sync, fragments -> Cs
      role: mma
      src: acc
      dst: C
      signal: [C.full]                                  # L135: __syncthreads
  - copy:                                               # L136-139: convert and store, bounds-checked
      role: load
      src: C
      dst: "Y[bm*tile.m : bm*tile.m + tile.m, bn*tile.n : bn*tile.n + tile.n]"
      cast: f16                                         # L138: __float2half
      mask: true                                        # L138: (m0 + r < M && n0 + c < N)
      wait: [C.full]                                    # L135: __syncthreads
```

The body is four top-level steps: the accumulator fill, one K-tile loop carrying both staged copies
and the matmul, the fragment-to-shared epilogue store, and the masked fp16 write. The loop covers
every K-tile including the first, so no prologue step is needed: `A[kt]` names stage `kt mod 2`,
`free` holds initially, and the sm86 lowering hoists the next stage's async copy and its group wait —
which is how the kernel's prologue and `wait_group 1` arise from a guard-free body (§8).

Mapping from the kernel to the document:

|kernel anchor|PKIR|
|---|---|
|L30-42 tile constants (`BM`/`BN`/`BK`/`STAGES`) and L36-37 `LDS`/`LDC` padding|`tile`, `buffers.A/B.stages`, `buffers.A/B.stride`, `buffers.C.stride`|
|L80-82 `__shared__` declarations; L90 accumulator fragments|`buffers` (`A`, `B`, `C`; `acc`)|
|L9-12 role comment and L57-75 `load_stage`|`roles.load`, the two `copy` steps (`pad: zero`, `async: true`)|
|L96-126 pipeline and MMA loop|the `for` step, `roles.mma`, the `matmul`, the slot lists|
|L111-124 warp grid and `kk` loop|`placement.sm86.warp_tile.mma` together with the device mma shape|
|L130-134 `store_matrix_sync`|`copy: {src: acc, dst: C}` signaling `C.full`|
|L136-139 storer loop and `__float2half`|`copy: {src: C, dst: "Y[...]", cast: f16, mask: true}` waiting `C.full`|
|L96-108 commit and `cp.async.wait_group`; L109, L125, L135 barriers|the `wait`/`signal` slot lists|
|L65 and L72 `ok` predicates|`pad: zero` on both loop copies|
|L151 `TORCH_CHECK(K % 8 == 0)`|`assume: ["K % 8 == 0"]`|
|L158-160 host grid|`grid.bm`, `grid.bn`|

Boundary semantics of the region: ragged `M`/`N` edges are zero-filled on the read and masked on the
write, so both are arbitrary. `K` is a positive multiple of 8. Inputs are fp16 and contiguous.

## §8 Lowering contract

Any lowering — CUDA now, NKI later — must preserve, for a validated region:

- every declared slot edge, emitted at least as strong as declared (no dropped barrier, no reordered
  publication);
- pads and masks never weakened: `pad: zero` stays a zero fill, `mask: true` stays a bounds check;
- buffer `stride` values, so the operand assumptions (`ldm` multiple, alignment) hold;
- capability ops realized by the §4 realization for the target device, or by a stronger one;
- `assume` predicates turned into host-side checks, as `TORCH_CHECK` does at L151.

A lowering may software-pipeline: the sm86 realization issues the `cp.async` for the next stage while
the current stage's MMAs run (L96-108) without changing the body, because the slot edges already state
the publication/consumption order that the hoist preserves.

For the worked example the CUDA realization's acceptance oracle is `tests/gpu/qkv_tile_kernel.cu`,
compared over the `SHAPES` table in `tests/gpu/test_qkv_tile.py` at `atol = rtol = 2e-2`, with
compilation, correctness and performance as separate admission criteria. The hardcoded
`-gencode=arch=compute_86,code=sm_86` in that test's `extra_cuda_cflags` is a property of the oracle
(RTX 3070), not of the IR.

## §9 Expansion points

Each item is a construct to add, not a wish: the widening contract for the rest of the fused
attention block. Until a construct lands, a region that needs it is out of v0.1 scope.

- **intra-tile reductions and broadcasts** with an explicit reduction scope and cross-thread combine
  tree — what RMSNorm's mean/variance and softmax's row max/sum need;
- **elementwise transcendental ops** with accuracy contracts (`exp`, `rsqrt`), so a lowering's
  approximation is an admitted property rather than an accident;
- **register-resident running state** for online softmax (max/sum carried across K-trips, with
  rescale), the construct that makes SDPA organizable as a region;
- **multi-region programs that share buffers** (fusion) with an equivalence obligation stated across
  region boundaries;
- **masks and predicates beyond rectangular tiles** (causal masking), extending §3's slice grammar;
- **layout transforms** (Q/K/V head split, transposes between regions) as first-class steps rather
  than as extra `copy` steps with implicit reindexing;
- **alignment padding for rows that are not 16-byte aligned**, which lifts `K % 8 == 0` from an
  `assume` to a padding construct;
- **a thread-map construct** for placement-level slice-partition proofs, so §5's thread-map obligation
  becomes statically checked instead of discharged by the differential test;
- **atomics and split-K reductions**;
- **capability queries for trn2**, replacing §4's `pending` column.

## §10 DeviceSpec interface obligations (I1)

PKIR v0.1 reads exactly these `DeviceSpec` keys. This is the interface the DeviceSpec and the
lowering agree on; PKIR asks for no other device knowledge, and absent keys take the DeviceSpec's
defaults.

|`DeviceSpec` key|what PKIR reads it for|sm86 (values the example exercises)|trn2|
|---|---|---|---|
|`memory_tiers` (names, capacity, `alignment_bytes`)|tier names referenced by `buffers.tier`, capacity for §6's footprint and occupancy check, alignment for slice and copy legality|`shared`, `register`; 16-byte alignment|pending — NKI task|
|`runtime` (`threads_per_warp`, `warps_per_block_max`)|thread count implied by `placement.warps`|32 threads/warp (4 warps = 128 threads, L34)|pending — NKI task|
|`copy_engines` (`chunk_bytes`, `element_bytes`, commit-group support, zero-fill-on-guard-false)|the chunk width a tile copy is realized in, whether `async: true` and its group waits exist, and whether `pad: zero` is the engine's own behavior|`chunk_bytes: 16`, `element_bytes: 2`, commit groups yes, zero-fill on guard-false yes (L44-47)|pending — NKI task|
|`capabilities.tensor_core_mma` (shapes, operand dtypes, layouts, `ldm` multiple, alignment)|divisibility of `warp_tile.mma` and validity of operand leading dimensions|`[16,16,16]`, f16 operands / f32 accumulate, A `row_major`, B `col_major`, `ldm` multiple 8 halves|pending — NKI task|
|`capabilities.convert` (dtype pairs, rounding)|the `cast` contract: one rounding, nearest-even|f32 → f16, round-to-nearest-even|pending — NKI task|
|`sync` (barrier scope, per-thread async wait)|realizing `wait`/`signal`: a barrier of the right scope plus the async engine's completion wait|block-scoped barrier (`__syncthreads()`, L109/L125/L135), per-thread group wait (`cp.async.wait_group`, L105-108)|pending — NKI task|
|`cost` (peak FLOPs per dtype, shared and register capacity)|§6's derived bytes, FLOPs and occupancy hint|shared capacity admits ≤2 blocks/SM for the example's 37,888 B/block|pending — NKI task|
