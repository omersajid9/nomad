# Running NOMAD on the GPU box

This runbook exercises the CUDA path end to end: a PyTorch linear region becomes a canonical
graph, a PKIR document, CUDA source, PTX, a loaded extension, a differential test and a timing
measurement. It assumes the RTX 3070/A40 host (`sm_86`) with an NVIDIA driver and a CUDA toolkit
whose `nvcc` matches the PyTorch CUDA build.

The full attention block is not expected to pass G1 yet: `norm`, `split`, `attn` and `merge` need
the PKIR v0.2 constructs. The QKV region is the v0 admitted path.

## 0. Host prerequisites

```sh
nvidia-smi
nvcc --version
python3 --version          # 3.11
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

The RTX 3070 and A40 are both `sm_86`. `src/nomad/devices/sm86.yaml` is calibrated for the A40;
timing is measured, so the architecture flag remains correct on the 3070.

## 1. Get the code

```sh
git clone git@github.com:omersajid9/nomad.git
cd nomad
git checkout main
git pull --ff-only
git log --oneline -5
```

The pipeline commit should be at or after the merge of `origin/hkohli/lower-cuda`.

## 2. Install and check the environment

```sh
uv sync
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

`cuda available` must be `True` before G4/G5 can run. `nvcc` must be on `PATH` or under
`$CUDA_HOME/bin` for G3 to emit PTX.

## 3. Run the CPU gates first

```sh
uv run pytest -m "not gpu and not nki" -q
uv run ruff check src tests experiments
uv run ruff format --check src tests experiments
```

This is the fast sanity check: parse, static, verifier and lowering-source tests.

## 4. Run the CUDA lowering tests

```sh
uv run pytest -m gpu tests/gpu/test_qkv_tile.py tests/gpu/test_lowered_qkv.py -q
```

`test_lowered_qkv.py` builds the extension emitted by
`src/nomad/lowering/cuda/tiled_gemm.py`, compares it to the hand-written oracle bit for bit, and
checks the `assume` host guard. The first build compiles a CUDA extension and can take about a
minute.

Optional timing table:

```sh
uv run python tests/gpu/test_lowered_qkv.py
```

## 5. Run the full QKV pipeline

```sh
RUN=results/runs/gpu-qkv
uv run nomad init qkv_proj "$RUN" --start unfused

# G0 parse, G1 static, G2 slot verification, G3 lower (+ PTX if nvcc is available)
uv run nomad check "$RUN" --upto G3

# Expected: admitted at lower.
# Artifacts:
#   $RUN/lowered/qkv_proj.cu
#   $RUN/lowered/qkv_proj.ptx
#   $RUN/lowered/qkv_proj.json

# Full stack: G0 -> G1 -> G2 -> G3 -> G4 difftest -> G5 timing
uv run nomad check "$RUN" --upto G5

# Record the result and print the selected snapshot.
uv run nomad submit "$RUN"
```

Expected G3 metadata:

```json
{
  "kernel": "qkv_proj",
  "shared_bytes": 37888,
  "threads": 128,
  "tile": [64, 64, 32],
  "warps": [2, 2]
}
```

Expected G4 rows: the three `qkv_v0` shapes `128x1024x3072`, `1000x1000x3000` and `1x64x64`.
Expected G5 metrics: median kernel time, `F.linear` time, per-shape speedup and
`geomean_speedup`.

## 6. Inspect the artifacts

```sh
cat "$RUN/diagnostics/latest.json"
cat "$RUN/lowered/qkv_proj.json"
sed -n '1,30p' "$RUN/lowered/qkv_proj.ptx"
ls -l "$RUN/lowered" "$RUN/snapshots"
```

The PTX header should say `.version 8.5`, `.target sm_86` and contain the `gemm_kernel` entry.
The admitted snapshot copies `lowered/` so the exact source, PTX and metadata that passed G5 stay
with the result.

## 7. Optional: run the agent loop

The driver speaks the OpenAI-compatible chat protocol and only accepts edits to `plan.yaml` and
`kernels/*.pkir.yaml`. Its defaults are OpenRouter and `deepseek/deepseek-v4-flash`; the key is
read from `OPENROUTER_API_KEY` and is never written to the workspace or a log.

```sh
export OPENROUTER_API_KEY=...   # your OpenRouter key; do not commit it
AGENT_RUN=results/runs/gpu-agent
uv run nomad init qkv_proj "$AGENT_RUN" --start unfused

uv run nomad agent "$AGENT_RUN" \
  --upto G1 \
  --max-steps 4 \
  --max-tokens 8192
```

To use a different OpenRouter model, pass `--model`; to use a local OpenAI-compatible endpoint,
pass both `--endpoint` and `--model` (and `--api-key` if it requires one).

For a repair run, first make the kernel invalid and let the model see the diagnostic:

```sh
python - "$AGENT_RUN" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1]) / "kernels" / "qkv_proj.pkir.yaml"
d = yaml.safe_load(p.read_text())
d["tile"]["bad"] = 1
p.write_text(yaml.safe_dump(d, sort_keys=False))
PY
uv run nomad check "$AGENT_RUN" --upto G1   # P-UNKNOWN-KEY
uv run nomad agent "$AGENT_RUN" --upto G1 --max-steps 4
```

The agent driver itself is independent of model quality. The local 2.9B reasoning model can
rewrite a scaffolded file but is unreliable on full-file kernel rewrites; a malformed response is
rejected by G1 and returned as a located diagnostic, never applied silently.

## 8. Full attention-block expectation

```sh
BLOCK=results/runs/gpu-block
uv run nomad init attention_block "$BLOCK" \
  --from-export tests/fixtures/export_block.py --start unfused
uv run nomad check "$BLOCK" --upto G1
```

Expected result: rejected at `static` with four `S-UNSUPPORTED` diagnostics for the `norm`,
`split`, `attn` and `merge` groups. The graph import works; the PKIR v0.2 constructs in
`docs/pkir-spec.md` §9 are required before those groups can lower.

## 9. Troubleshooting

| symptom | meaning | action |
|---|---|---|
| `D-UNAVAILABLE` | `torch.cuda.is_available()` is false | check the driver installation and that `uv run python` sees the GPU |
| `L-TOOLCHAIN` | no `nvcc` found | add the CUDA toolkit to `PATH`, or set `CUDA_HOME` |
| `L-COMPILE` | generated CUDA failed to compile | read `lowered/GROUP.ptx.cu`; run `nvcc -ptx -arch=sm_86 -O3 -std=c++17` manually |
| `D-CRASH` | extension build or kernel launch failed | re-run with the extension build log; check `nvcc --version` against `torch.version.cuda` |
| `S-CAPACITY` / `L-RESOURCE` | tile/stage footprint is too large | reduce `stages`, `tile.m` or `tile.n` in the PKIR document |
| `S-UNSUPPORTED` on the block | expected PKIR v0.2 gap | only the linear groups are in v0 scope |
| `S-SLICE` after an agent edit | the model returned a bad slice | the diagnostic is localized; repair the file or reset from `--start unfused` |
| `401 User not found` from OpenRouter | the key is a provisioning/management key, not an inference key | create an inference key in OpenRouter and put it in `OPENROUTER_API_KEY` |

## 10. What to record

For the report, keep:

1. the GPU model, driver version, `nvcc --version`, `torch.version.cuda`;
2. the G3 metadata (`shared_bytes`, `threads`, `tile`, `warps`);
3. the G4 shape table and all `max_abs_err` values;
4. the G5 per-shape times and `geomean_speedup`;
5. the agent step count, token total and whether the run was admitted or failed;
6. the full-block `S-UNSUPPORTED` result as the scope boundary, not as a pipeline failure.

Do not compare the A40-calibrated cost model to measured 3070 timing without saying which device
produced each number.
