# NOMAD

A device-agnostic, verifier-in-the-loop agentic kernel compiler.

An agent proposes graph- and kernel-level schedules over a Portable Kernel IR (PKIR); an ordered
gate stack — static checks → verification → lowering → differential tests → timing — admits them
with localized diagnostics. Only the `DeviceSpec` changes between NVIDIA `sm86` and Trainium
`trn2`. Target scope: one fused attention block (RMSNorm → QKV → SDPA with online softmax →
out-proj, fp16).

CS 521 (Fall 2026) final project — Hemank Kohli, Omer Sajid.
Proposal: [docs/proposal.pdf](docs/proposal.pdf).

**Status: spec and references landed.** PKIR v0.1 is specified in [docs/pkir-spec.md](docs/pkir-spec.md)
with the I1 QKV projection document as its worked example, shipped for tooling as
[`examples/qkv_proj.pkir.yaml`](examples/qkv_proj.pkir.yaml); the annotated reference store is in
[docs/refs/](docs/refs/index.md). Implementation, device specs and schemas have not landed yet.

## Layout

| Path | Contents |
| --- | --- |
| `src/nomad/devicespec/` | DeviceSpec models, loader, capability queries |
| `src/nomad/devices/` | Canonical device YAMLs (v0: `sm86`, `trn2`), overridable via `--specs-dir` |
| `src/nomad/pkir/` | PKIR ops/types, builder, parser/printer, type + capability checker |
| `src/nomad/verifier/` | Memory safety, sync (race/deadlock) freedom, symbolic equivalence |
| `src/nomad/lowering/` | Template lowerings — `cuda/` and `nki/`; all device-specific code lives here |
| `src/nomad/graph/` | Agent action space: fusion plans, layout set, gated rewrites |
| `src/nomad/cost/` | Spec-driven cost model |
| `src/nomad/reference/` | PyTorch reference for the fused attention block |
| `src/nomad/difftest/` | Differential-test harness (reference vs. lowered kernels) |
| `src/nomad/harness/` | `nomad` CLI: ordered gates, diagnostics, agent driver |
| `schemas/` | JSON Schema for the agent contracts (specs, plans, diagnostics) |
| `examples/` | Agent-facing example plans and PKIR programs |
| `tests/` | `unit/`, `differential/`, `gpu/`, `nki/`, `fixtures/` |
| `mutants/` | ≈10 injected bugs for the gate study, with manifest |
| `experiments/` | `latency/`, `portability/`, `ablation/`, `bug_injection/` runners + configs |
| `results/` | Raw run output (gitignored; curated tables land under `experiments/`) |
| `env/` | Machine setup: CUDA box (Linux + nvcc), Neuron box (Linux x86_64) |
| `docs/` | Proposal, PKIR spec, DeviceSpec notes, integration points (I1–I4) |
| `report/` | ASPLOS-style 5-page report |

## Quickstart

```sh
uv sync                                   # Python 3.11 + deps (torch included)
uv run pytest -m "not gpu and not nki"    # CPU-only subset, what CI runs
uv run ruff check . && uv run ruff format --check .
```

Python stays at 3.11 (safe intersection of the Neuron SDK and PyTorch); `uv.lock` is committed.

## Conventions

- Tests are marked `gpu` (needs the RTX 3070 box) and `nki` (needs a Linux x86_64 Neuron
  environment); neither runs on a laptop or in CI.
- Workstreams: HK — CUDA path, agent driver, GPU experiments. OS — DeviceSpec/PKIR, verifier,
  NKI path. Integration points I1–I4 are pair-programmed; their interfaces land in `docs/`.
- Commits: `[scope] imperative summary`, one logical change per commit.
