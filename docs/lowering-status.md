# Pipeline status: QKV end to end, full block still PKIR v0.2

**Updated 2026-11-15** after merging `origin/hkohli/lower-cuda` and wiring the gate stack.

## What now works

| stage | state |
|---|---|
| PyTorch → `graph.yaml` | one bias-free linear via `--from-export`; the reference attention block is recognized structurally and mapped to the canonical graph |
| `graph.yaml` → plan/catalog | partition, pattern, order, layout and rewrite checks (G1) |
| plan → PKIR | scaffolded GEMM template; agent edits `plan.yaml` and `kernels/*.pkir.yaml` |
| PKIR → CUDA | `src/nomad/lowering/cuda/tiled_gemm.py` emits the tiled-GEMM extension source |
| CUDA → PTX | `nomad lower --ptx` / G3 when `nvcc` is on PATH; device-only source compiles with `nvcc -ptx` |
| G2 | slot-discipline verifier (`V-*`) checks async/full/free and epilogue edges |
| G4 | builds the lowered extension and compares each shape to `F.linear`; `D-*` diagnostics |
| G5 | CUDA-event timing and speedup; cost estimate on hosts without hardware |
| agent | `nomad agent` drives an OpenAI-compatible model through the file-update protocol |

The QKV region is admitted through **G3 on a CPU checkout** (source generated; PTX if `nvcc`
exists). **G4/G5 require the `sm86` GPU box**; on this Mac G4 returns `D-UNAVAILABLE`, not a fake
pass.

## Verified here

```text
uv run pytest -m "not gpu and not nki"  ->  71 passed
G2 (canonical qkv)                      ->  admitted
G3 (canonical qkv)                      ->  lowered/qkv_proj.cu, 37,888 B shared, 128 threads
G4 (this Mac)                           ->  D-UNAVAILABLE (no CUDA runtime)
generated device source                 ->  nvcc 12.6.3 -arch=sm_86 -ptx succeeds (22,933 B PTX)
```

The PTX was compiled in the `nvidia/cuda:12.6.3-devel-ubuntu22.04` container; no GPU was needed
for `nvcc -ptx`.

## Missing, by scope

1. **Full attention-block lowering.** `rmsnorm`, `split`, `attn`, `merge` remain `S-UNSUPPORTED`.
   They need PKIR v0.2 constructs from `docs/pkir-spec.md §9`: reductions/broadcasts, `exp`/`rsqrt`,
   running softmax state, layout transforms and causal masks. The export and plan layers now accept
   the block graph, but G1 stops before G3.
2. **Symbolic equivalence (G2 v1).** The current verifier checks slot discipline, not region
   equivalence. G4 supplies the numeric oracle check.
3. **NKI lowering.** `src/nomad/lowering/nki/` is a stub; `trn2` is pending.
4. **GPU validation.** G4/G5, the oracle comparison and timing have not run on the RTX 3070/A40
   from this machine. The test `tests/gpu/test_pipeline_gpu.py` is ready for that box.
5. **Measured block latency and ablations.** The pipeline can measure the QKV group; the proposal's
   block-level latency, portability and ± gates ablations await full-block lowering.
6. **Agent reliability.** The local 2.9B reasoning model completes a scaffold edit/repair, but
   its full-file rewrite of the GEMM template produced an invalid slice in a manual trial. The
   driver correctly rejected it and preserved diagnostics; this is a model/protocol limitation,
   not a gate bypass.
