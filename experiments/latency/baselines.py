"""Block latency of the PyTorch baselines (eager, SDPA, torch.compile) on the GPU box.

uv run python experiments/latency/baselines.py > results/latency_baselines.csv
"""

import sys
from functools import partial

import torch

from nomad.difftest.harness import compare, time_ms
from nomad.reference.block import BASELINES, SHAPES, make_inputs, reference_fp32


def main() -> None:
    gpu = torch.cuda.get_device_name()
    print(gpu, "| torch", torch.__version__, "| CUDA", torch.version.cuda, file=sys.stderr)
    print("shape,baseline,ms,max_abs_err,ok")
    for shape in SHAPES:
        x, p = make_inputs(shape)
        ref = reference_fp32(x, p, shape)
        for name, fn in BASELINES.items():
            r = compare(fn(x, p, shape), ref, shape.id)  # also triggers torch.compile's first build
            ms = time_ms(partial(fn, x, p, shape))
            print(f"{shape.id},{name},{ms:.4f},{r.max_abs_err:.3e},{r.ok}", flush=True)


if __name__ == "__main__":
    main()
