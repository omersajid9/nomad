"""The CUDA lowering on CPU: what it reads from a PKIR document and what it emits (runs in CI)."""

import copy
from pathlib import Path

import pytest
import yaml

from nomad.lowering.common import LoweringError
from nomad.lowering.cuda.tiled_gemm import lower, match

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "qkv_proj.pkir.yaml"


@pytest.fixture
def doc() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def loop_step(doc: dict, i: int) -> dict:
    return next(iter(doc["body"][1]["for"]["body"][i].values()))


def lowering_error(doc: dict) -> LoweringError:
    with pytest.raises(LoweringError) as info:
        lower(doc)
    return info.value


def test_the_example_reads_the_oracles_constants(doc: dict) -> None:
    r = match(doc)
    assert (r.tile, r.stages, r.warps, r.threads) == ((64, 64, 32), 2, (2, 2), 128)
    assert (r.lda, r.ldb, r.ldc) == (40, 40, 68)
    assert r.shared_bytes == 37_888  # the worked number in docs/pkir-spec.md §6


def test_the_example_realizes_every_declared_edge_pad_mask_and_assume(doc: dict) -> None:
    src = lower(doc).source
    assert src.count("__syncthreads();") == 3  # operand full, operand free, epilogue full
    assert src.count("cp_async_wait<STAGES - 1>();") == 1
    assert src.count("const bool ok") == 2
    assert "if (m0 + r < M && n0 + c < N)" in src
    assert 'TORCH_CHECK((K % 8) == 0, "assume violated: K % 8 == 0");' in src
    assert "const int k_tiles = ((K + BK - 1) / BK);" in src


def test_lowering_is_deterministic(doc: dict) -> None:
    assert lower(doc) == lower(copy.deepcopy(doc))


# Mutants: the lowering realizes what is declared, so a weakened document gives a weaker kernel.
def test_dropping_the_free_waits_drops_the_trailing_barrier(doc: dict) -> None:
    for i in (0, 1):
        del loop_step(doc, i)["wait"]
    assert lower(doc).source.count("__syncthreads();") == 2


def test_dropping_the_full_wait_drops_the_copy_wait_and_barrier(doc: dict) -> None:
    del loop_step(doc, 2)["wait"]
    src = lower(doc).source
    assert "cp_async_wait<" not in src.split("load_stage(&As[kn")[1]
    assert src.count("__syncthreads();") == 2


def test_dropping_pad_drops_the_zero_fill_guard(doc: dict) -> None:
    del loop_step(doc, 0)["pad"]
    assert lower(doc).source.count("const bool ok") == 1


def test_dropping_the_mask_stores_unguarded(doc: dict) -> None:
    del doc["body"][3]["copy"]["mask"]
    assert "if (m0 + r < M" not in lower(doc).source


def test_a_skipped_k_tile_is_lowered_faithfully(doc: dict) -> None:
    doc["body"][1]["for"]["trips"] = "ceil_div(K, tile.k) - 1"
    assert "const int k_tiles = (((K + BK - 1) / BK) - 1);" in lower(doc).source


def test_sync_copies_need_no_cp_async(doc: dict) -> None:
    for i in (0, 1):
        del loop_step(doc, i)["async"]
    src = lower(doc).source
    assert "cp_async_16(As_st" not in src and "cp_async_wait<STAGES" not in src
    assert src.count("make_uint4(0, 0, 0, 0)") == 2


# Limits: L-RESOURCE is the agent's to fix; L-TEMPLATE names the field the template cannot realize.
def test_three_stages_fit_in_static_shared_memory(doc: dict) -> None:
    for name in ("A", "B"):
        doc["buffers"][name]["stages"] = 3  # 48,128 B
    assert "constexpr int STAGES = 3;" in lower(doc).source


def test_four_stages_are_a_resource_error(doc: dict) -> None:
    for name in ("A", "B"):
        doc["buffers"][name]["stages"] = 4  # 58,368 B > 48 KiB static
    assert lowering_error(doc).code == "L-RESOURCE"


def test_a_non_transposed_weight_is_a_template_limit(doc: dict) -> None:
    loop_step(doc, 2)["transpose_b"] = False
    err = lowering_error(doc)
    assert (err.code, err.pointer) == ("L-TEMPLATE", "/body/1/for/body/2/matmul/transpose_b")


def test_a_non_canonical_slice_names_its_field(doc: dict) -> None:
    loop_step(doc, 0)["src"] = "X[bm*tile.m : bm*tile.m + tile.m, 0 : tile.k]"
    assert lowering_error(doc).pointer == "/body/1/for/body/0/copy/src"


def test_a_warp_tile_off_the_mma_shape_names_its_field(doc: dict) -> None:
    doc["placement"]["sm86"]["warp_tile"]["mma"] = ["tile.m / 2", "tile.n / 8"]  # 8 columns
    assert lowering_error(doc).pointer == "/placement/sm86/warp_tile/mma"


def test_a_misshapen_document_is_a_template_error_not_a_crash(doc: dict) -> None:
    del doc["buffers"]
    assert lowering_error(doc).code == "L-TEMPLATE"