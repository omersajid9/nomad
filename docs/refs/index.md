# Kernel-compiler reference store

Annotated bibliography for NOMAD: the papers whose design decisions PKIR v0.1 adopts, each with the
spec anchors it supports and a local PDF where one exists, so the CUDA box and the Neuron box can
read them offline.

*Links and PDFs fetched 2026-11-01. The ACM DL links (`triton`, `taso`, `tensorir`, `gpuverify`)
return 403 to non-browser clients and were metadata-verified through their Crossref DOI records
instead; every title, author list and year below was fetched, not recalled.*

## Papers

### cake

```text
citation:  CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution — Zihao Ye, Yingyi Huang, et al. — arXiv:2608.12629 (2026)
link:      https://arxiv.org/abs/2608.12629
pdf:       pdfs/cake.pdf
why:       CAKE co-designs a typed, hardware-explicit schedule IR with the agent loop that rewrites kernels, and feeds verification results back to the agent as localized diagnostics. PKIR v0.1 takes that posture: roles, staged pipelines and capability-gated ops in the body (§3), and per-construct semantic contracts the harness hands back as located errors (§4).
spec:      docs/pkir-spec.md §3, §4
```

### kernelbench

```text
citation:  KernelBench: Can LLMs Write Efficient GPU Kernels? — Anne Ouyang, Simon Guo, et al. — arXiv:2502.10517 (2025)
link:      https://arxiv.org/abs/2502.10517 + https://scalingintelligence.stanford.edu/blogs/kernelbench/
pdf:       pdfs/kernelbench.pdf
why:       KernelBench makes compilation, correctness and speedup three separate criteria, each gating the next, over a fixed set of shapes with a 1e-2 tolerance. §5 keeps exactly that separation — structural slice/slot checks, then equivalence, then timing — and §1 scopes PKIR to the part of the stack those gates consume.
spec:      docs/pkir-spec.md §1, §5
```

### kernelbench-verified

```text
citation:  KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch? — Yunxiang Zhang, Ping Yu, et al. — arXiv:2607.16241 (2026)
link:      https://arxiv.org/abs/2607.16241
pdf:       pdfs/kernelbench-verified.pdf
why:       Re-auditing benchmark-passing kernels under stricter checks (hidden distributions, careful baselines) shows that a passing kernel can still be wrong. That is why §5 states the equivalence obligation per construct — zero padding contributes nothing, the store is bounds-checked, accumulation is fp32, the output is rounded once — instead of trusting a naive allclose, and why §8 fixes the tolerance as part of the oracle rather than as a knob.
spec:      docs/pkir-spec.md §5, §8
```

### tritorx

```text
citation:  Agentic Operator Generation for ML ASICs — Alec M. Hammond, Aram Markosyan, et al. (Meta/FAIR) — arXiv:2512.10977 (2025)
link:      https://arxiv.org/abs/2512.10977
pdf:       pdfs/tritorx.pdf
why:       TritorX drives generation with a linter, JIT compilation and a test harness in a finite state machine, targeting an accelerator behind an abstraction rather than one vendor's ISA. §2's limits-first document rules (unknown keys are schema errors, every failure reports its YAML line) and §5's localized defect reporting are the PKIR-side form of that same feedback loop.
spec:      docs/pkir-spec.md §2, §5
```

### tritonbench

```text
citation:  TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators — Jianling Li et al. — arXiv:2502.14752 (2025)
link:      https://arxiv.org/abs/2502.14752
pdf:       pdfs/tritonbench.pdf
why:       A differential-test harness organized as a shape table per operator, ragged and edge shapes included. §5 and §8 use the same convention: the worked example's acceptance oracle is the SHAPES table in tests/gpu/test_qkv_tile.py, whose ragged rows are M = 1, M and N not multiples of the tile, and K a multiple of 8 but not of 32.
spec:      docs/pkir-spec.md §5, §8
```

### proofwright

```text
citation:  ProofWright: Towards Agentic Formal Verification of CUDA — Bodhisatwa Chatterjee, Drew Zagieboylo, et al. — arXiv:2511.12294 (2025)
link:      https://arxiv.org/abs/2511.12294
pdf:       pdfs/proofwright.pdf
why:       ProofWright closes the loop between kernel synthesis and formal verification of the memory accesses and synchronization of the generated CUDA. §5's slice safety and slot discipline are the structural form of those obligations, discharged on the region itself instead of on an offline annotation pass over lowered code.
spec:      docs/pkir-spec.md §5
```

### tilelang

```text
citation:  TileLang: A Composable Tiled Programming Model for AI Systems — Lei Wang et al. — arXiv:2504.17577 (2025)
link:      https://arxiv.org/abs/2504.17577
pdf:       pdfs/tilelang.pdf
why:       TileLang separates dataflow (tiles, buffers, pipelined stages) from its realization (thread binding, layouts, tensorization, software pipelining). PKIR splits along the same seam: §3's buffers and body carry no device numbers, and placement holds every one of them, which is what makes §3's fallback rule meaningful — the same body lowers unchanged for a device whose placement block is empty.
spec:      docs/pkir-spec.md §3
```

### dnnfusion

```text
citation:  DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion — Wei Niu et al. — arXiv:2108.13342 (2021)
link:      https://arxiv.org/abs/2108.13342
pdf:       pdfs/dnnfusion.pdf
why:       DNNFusion systematizes operator fusion with a taxonomy of fusion types, the graph-level layer the agent acts on. It sits above PKIR: §1 puts fusion plans in the action space above the IR, and §9's multi-region construct is the IR-side counterpart a fused plan compiles into, with the equivalence obligation stated across region boundaries.
spec:      docs/pkir-spec.md §1, §9
```

### mlir

```text
citation:  MLIR: A Compiler Infrastructure for the End of Moore's Law — Chris Lattner et al. — arXiv:2002.11054 (2020)
link:      https://arxiv.org/abs/2002.11054
pdf:       pdfs/mlir.pdf
why:       MLIR's layering argument — small dialectable IRs with explicit extension points instead of one universal IR — is why PKIR stays device-free and lets device instructions enter only as capability-gated ops. §2's extension discipline (one document form, unknown keys are schema errors, a canonical expression printer) is what keeps that layering from eroding as §9 widens.
spec:      docs/pkir-spec.md §2
```

### flashattention

```text
citation:  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Tri Dao, Daniel Y. Fu, et al. — arXiv:2205.14135 (2022)
link:      https://arxiv.org/abs/2205.14135
pdf:       pdfs/flashattention.pdf
why:       FlashAttention shows that attention is a memory-hierarchy tiling problem, keeping a running softmax state in registers and masking the padded columns out of the row max and sum. §9's register-resident running state (max/sum with rescale) and its masking construct are exactly what PKIR must add before SDPA is organizable as a region; v0.1 stops short on purpose.
spec:      docs/pkir-spec.md §9
```

### flashattention2

```text
citation:  FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning — Tri Dao — arXiv:2307.08691 (2023)
link:      https://arxiv.org/abs/2307.08691
pdf:       pdfs/flashattention2.pdf
why:       FlashAttention-2's gain comes from how the work is partitioned across warps and blocks, with the numerics of v1 unchanged. That is placement, not body: two placements of one region must be numerically identical, which is why §3 confines every device number to placement and §5 makes "the placement's thread map partitions each tile move" an explicit obligation discharged by the differential test.
spec:      docs/pkir-spec.md §3, §5
```

### triton

```text
citation:  Triton: an intermediate language and compiler for tiled neural network computations — Tillet, Kung, Cox — MAPL@PLDI 2019; DOI:10.1145/3315508.3329973
link:      https://doi.org/10.1145/3315508.3329973
pdf:       — (link-only)
why:       Triton introduced the tile-space program — operations over blocks of a tensor with the thread mapping left to the compiler. §3's tile-shaped buffers, tile-size symbols and role-annotated steps are PKIR's version of that prior art, with the mma warp tile kept in placement so the body stays device-agnostic.
spec:      docs/pkir-spec.md §3
```

### taso

```text
citation:  TASO: optimizing deep learning computation with automatic generation of graph substitutions — Jia et al. — SOSP 2019; DOI:10.1145/3341301.3359630
link:      https://doi.org/10.1145/3341301.3359630
pdf:       — (link-only)
why:       TASO generates candidate graph substitutions and verifies them before applying them, rather than trusting a hand-written rewrite set. §9's gated rewrites reuse that structure one level down: an agent-proposed region is admitted only after it passes §5's obligations and the differential test.
spec:      docs/pkir-spec.md §9
```

### tensorir

```text
citation:  TensorIR: An Abstraction for Automatic Tensorized Program Optimization — Siyuan Feng, Bohan Hou, et al. — ASPLOS 2023; DOI:10.1145/3575693.3576933 (arXiv:2207.04296)
link:      https://arxiv.org/abs/2207.04296 + https://doi.org/10.1145/3575693.3576933
pdf:       — (link-only)
why:       TensorIR makes tensorized schedules first-class IR — blocks, buffers, layouts and the loop structure over them — so that schedule transformations are mechanical rather than ad hoc. §3's tile/buffers/affine-slice trio is PKIR's deliberately smaller counterpart, and §9's thread-map construct is the piece v0.1 leaves to placement obligations instead of encoding.
spec:      docs/pkir-spec.md §3, §9
```

### gpuverify

```text
citation:  GPUVerify: a verifier for GPU kernels — Betts, Chong, Donaldson, et al. — OOPSLA 2012; DOI:10.1145/2384616.2384625
link:      https://doi.org/10.1145/2384616.2384625
pdf:       — (link-only)
why:       GPUVerify established barrier-aware, thread-modular reasoning for GPU kernels: data races and barrier divergence decided against a declared synchronization structure. §5's slot discipline — every read waits full, every write waits free, every participant of a role reaches each wait and signal — is that obligation in slot form, checked structurally on the region rather than on lowered CUDA.
spec:      docs/pkir-spec.md §5
```

## Toolchain references

- PTX ISA — https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA C++ Programming Guide — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html (its Warp Matrix Functions section is what the tensor-core row of the spec's §4 is checked against)
- CUDA Ampere tuning guide — https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- Nsight Compute — https://docs.nvidia.com/nsight-compute/
- Triton docs — https://triton-lang.org/main/index.html
- TileLang repository — https://github.com/tile-ai/tilelang
- AWS NKI samples — https://github.com/aws-neuron/nki-samples
