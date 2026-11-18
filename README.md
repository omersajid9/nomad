# NOMAD

A device-agnostic, verifier-in-the-loop agentic kernel compiler.

An agent proposes graph- and kernel-level schedules over a Portable Kernel IR (PKIR); an ordered
gate stack — static checks → verification → lowering → differential tests → timing — admits them
with localized diagnostics. Only the `DeviceSpec` changes between NVIDIA `sm86` and Trainium
`trn2`. Target scope: one fused attention block (RMSNorm → QKV → SDPA with online softmax →
out-proj, fp16).

CS 521 (Fall 2026) final project — Hemank Kohli, Omer Sajid.
Proposal: [docs/proposal.pdf](docs/proposal.pdf).

**Status: the QKV region runs through the ordered gate stack.** PyTorch `torch.export` (or the
built-in canonical graph) -> `graph.yaml` -> plan/catalog -> PKIR -> CUDA lowering -> PTX ->
difftest -> timing. `nomad init/check/cost/scaffold/lower/explain/submit` implements the agent
contract's workspace; `nomad agent` drives any OpenAI-compatible model through the file-update
loop. G0-G3 (parse, static, slot verification, lowering) run on CPU and write
`lowered/GROUP.cu`; G4 difftest and G5 timing need the `sm86` GPU box. `nvcc` additionally emits
`lowered/GROUP.ptx` during G3. The full attention block now imports from `torch.export`, but its
RMSNorm/SDPA/layout regions remain `S-UNSUPPORTED` until the PKIR v0.2 constructs in
[docs/pkir-spec.md §9](docs/pkir-spec.md) land; only the linear regions lower.

## Layout

| Path | Contents |
| --- | --- |
| `src/nomad/devicespec/` | DeviceSpec models, loader, capability queries |
| `src/nomad/devices/` | Canonical device YAMLs (`sm86` calibrated for A40; `trn2` pending) |
| `src/nomad/pkir/` | PKIR pydantic models, expression/slice parser, static checker |
| `src/nomad/verifier/` | G2 slot-discipline verifier (memory/sync/equivalence reserved) |
| `src/nomad/lowering/` | PKIR→CUDA template lowering; `nki/` remains a stub |
| `src/nomad/graph/` | PyTorch export (linear and attention block), plan/catalog checks, rewrites, layouts |
| `src/nomad/cost/` | Spec-driven analytic cost model (`nomad cost`) |
| `src/nomad/reference/` | PyTorch reference block and canonical graph documents |
| `src/nomad/difftest/` | Differential-test harness and G4 group adapter |
| `src/nomad/harness/` | `nomad` CLI: workspace, G0-G5 gates, lowering artifacts, diagnostics |
| `src/nomad/agent/` | Agent driver, file-update protocol, OpenAI-compatible client |
| `schemas/` | JSON Schema for the agent contracts (pydantic models are the source; emission pending) |
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

# QKV pipeline
uv run nomad init qkv_proj results/runs/001                  # default: blank kernel, agent authors it
uv run nomad init qkv_proj results/runs/002 --start unfused  # scaffolded baseline (contract §8 start)
uv run nomad init qkv_proj results/runs/003 --from-export tests/fixtures/export_qkv.py --start unfused
uv run nomad check results/runs/001 --upto G3   # G0/G1/G2 + CUDA source (PTX if nvcc is present)
uv run nomad lower results/runs/002             # lowered/qkv_proj.cu (+ .ptx with --ptx)
uv run nomad check results/runs/002 --upto G5   # G4 difftest and G5 timing on the GPU box
uv run nomad cost results/runs/001 --group qkv_proj
uv run nomad explore results/runs/002          # feasible rewrite subsets x orders
uv run python experiments/single_pass.py --task qkv_proj --run results/runs/pass-qkv

# Drive the agent through OpenRouter (default model: deepseek/deepseek-v4-flash)
export OPENROUTER_API_KEY=...   # never commit the key
uv run nomad agent results/runs/001 --upto G1 --max-steps 4
```

Python stays at 3.11 (safe intersection of the Neuron SDK and PyTorch); `uv.lock` is committed.

## Conventions

- Tests are marked `gpu` (needs the RTX 3070 box) and `nki` (needs a Linux x86_64 Neuron
  environment); neither runs on a laptop or in CI.
- Workstreams: HK — CUDA path, agent driver, GPU experiments. OS — DeviceSpec/PKIR, verifier,
  NKI path. Integration points I1–I4 are pair-programmed; their interfaces land in `docs/`.
- Rewrites are order-sensitive: `plan.yaml.rewrites` is an ordered list, and a rule applies only
  when its preconditions hold after the earlier rules. `nomad explore` enumerates the feasible
  subsets and orders for the current graph; the agent picks one sequence and `nomad check`
  validates it in order (`S-REWRITE` names the first rule that cannot apply).
- Commits: `[scope] imperative summary`, one logical change per commit.
