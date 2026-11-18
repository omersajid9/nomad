"""PKIR -> CUDA template lowering for tiled-GEMM regions (the QKV and out-proj projections).

`lower(doc)` matches a PKIR v0.1 document against the tiled-GEMM template and emits the CUDA C++ of
a torch extension. The template realizes what the document declares and nothing more
(docs/pkir-spec.md §8): a slot edge becomes a barrier only where a step waits on it, `pad: zero`
becomes the zero-filling copy predicate, `mask: true` the bounds-checked store, and `assume` host
checks. Mutating the document therefore mutates the kernel, which the gate study relies on.

For examples/qkv_proj.pkir.yaml the output is tests/gpu/qkv_tile_kernel.cu with its constants read
from the document, so the two kernels must agree bit for bit.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from string import Template
from typing import Any

from nomad.devicespec.spec import DeviceSpec
from nomad.lowering.common import LoweringError, const_int, to_cpp


@dataclass(frozen=True)
class CudaTarget:
    """The DeviceSpec keys this template reads (pkir-spec §10)."""

    name: str = "sm86"
    arch: str = "86"
    threads_per_warp: int = 32
    max_threads_per_block: int = 1024
    static_shared_bytes: int = 48 * 1024
    chunk_bytes: int = 16  # copy_engines.chunk_bytes; cp.async.cg moves exactly 16 bytes
    mma: tuple[int, int, int] = (16, 16, 16)  # capabilities.tensor_core_mma
    ldm_multiple_f16: int = 8
    ldm_multiple_f32: int = 4

    @classmethod
    def from_device(cls, device: DeviceSpec) -> CudaTarget:
        """Read the sm86 lowering target from a DeviceSpec, or reject a non-CUDA device."""
        if not device.arch.startswith("sm"):
            raise LoweringError(
                "L-TARGET", "", f"{device.name!r} is not an sm86-family CUDA device"
            )
        try:
            arch = device.arch.removeprefix("sm")
            mma = tuple(device.capabilities.tensor_core_mma.shapes[0])
            shared = device.memory_tiers["shared"].max_per_block_bytes
            chunk = device.copy_engines.chunk_bytes
            ldm = device.capabilities.tensor_core_mma.ldm_multiple
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LoweringError("L-TARGET", "", f"incomplete DeviceSpec: {exc}") from exc
        if len(mma) != 3:
            raise LoweringError("L-TARGET", "", "tensor_core_mma.shapes[0] must have three extents")
        return cls(
            name=device.name,
            arch=arch,
            threads_per_warp=device.runtime.threads_per_warp,
            max_threads_per_block=device.runtime.threads_per_block_max,
            static_shared_bytes=min(48 * 1024, shared or 48 * 1024),
            chunk_bytes=chunk,
            mma=(int(mma[0]), int(mma[1]), int(mma[2])),
            ldm_multiple_f16=ldm,
            ldm_multiple_f32=4,
        )


SM86 = CudaTarget()

_LOOP = "/body/1/for/body"


@dataclass(frozen=True)
class GemmRegion:
    """What the template needs from a matched document."""

    kernel: str
    tile: tuple[int, int, int]  # (tile.m, tile.n, tile.k)
    stages: int
    lda: int
    ldb: int
    ldc: int
    warps: tuple[int, int]  # warp grid (along M, along N)
    fill: float
    async_copy: tuple[bool, bool]  # per operand copy (A, B)
    pad: tuple[bool, bool]
    mask: bool
    full_edge: bool  # the matmul waits on an operand `full` slot
    free_edge: bool  # an operand copy waits on its `free` slot
    out_edge: bool  # the output store waits on the epilogue buffer's `full` slot
    trips: str  # C++
    grid: tuple[str, str]  # C++ (along M, along N)
    assume: tuple[tuple[str, str], ...]  # (C++ predicate, PKIR text)

    @property
    def threads(self) -> int:
        return self.warps[0] * self.warps[1] * SM86.threads_per_warp

    @property
    def shared_bytes(self) -> int:
        bm, bn, _ = self.tile
        return 2 * self.stages * (bm * self.lda + bn * self.ldb) + 4 * bm * self.ldc


def _expect(ok: Any, pointer: str, message: str, code: str = "L-TEMPLATE") -> None:
    if not ok:
        raise LoweringError(code, pointer, message)


def _step(steps: list, i: int, kind: str, base: str) -> dict:
    step = steps[i]
    _expect(isinstance(step, Mapping) and list(step) == [kind], f"{base}/{i}", f"expected `{kind}`")
    return step[kind]


def _flat(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def _slots(step: Mapping, key: str) -> set[str]:
    return set(step.get(key) or [])


def match(doc: Mapping, target: CudaTarget = SM86) -> GemmRegion:
    """Read a tiled-GEMM region from a PKIR document, or raise LoweringError where it differs."""
    _expect(doc.get("pkir") == 0, "/pkir", "the template reads PKIR revision 0")
    kernel = str(doc.get("kernel", ""))
    _expect(re.fullmatch(r"[A-Za-z_]\w*", kernel), "/kernel", "must be a C identifier")
    tile = {k: const_int(v, {}, f"/tile/{k}") for k, v in doc["tile"].items()}
    _expect(set(tile) == {"m", "n", "k"}, "/tile", "expected tile.m, tile.n, tile.k")
    bm, bn, bk = tile["m"], tile["n"], tile["k"]

    body = doc["body"]
    _expect(len(body) == 4, "/body", "expected fill, for, copy, copy")
    fill = _step(body, 0, "fill", "/body")
    loop = _step(body, 1, "for", "/body")
    to_shared = _step(body, 2, "copy", "/body")
    store = _step(body, 3, "copy", "/body")
    inner = loop["body"]
    _expect(len(inner) == 3, _LOOP, "expected copy, copy, matmul")
    loads = [_step(inner, i, "copy", _LOOP) for i in (0, 1)]
    mm = _step(inner, 2, "matmul", _LOOP)
    kt, mmp = loop["var"], f"{_LOOP}/2/matmul"

    # Operands: the matmul names the staged buffers; each copy into one of them is its loader.
    operands = []
    for key in ("a", "b"):
        ref = re.fullmatch(rf"(\w+)\[{re.escape(kt)}\]", _flat(mm[key]))
        _expect(ref, f"{mmp}/{key}", f"expected a staged buffer indexed by {kt}")
        operands.append(ref.group(1))
    a_buf, b_buf = operands
    by_dst = {_flat(c["dst"]): (i, c) for i, c in enumerate(loads)}
    _expect(set(by_dst) == {f"{a_buf}[{kt}]", f"{b_buf}[{kt}]"}, _LOOP, "one copy per operand")
    (ia, load_a), (ib, load_b) = by_dst[f"{a_buf}[{kt}]"], by_dst[f"{b_buf}[{kt}]"]
    _expect(mm.get("transpose_b") is True, f"{mmp}/transpose_b", "only W[N, K] operands")
    acc = mm["acc"]

    # Tensors: X[M, K] and W[N, K] in, Y[M, N] out, all f16, shapes over bare params.
    x_name = _flat(load_a["src"]).split("[")[0]
    w_name = _flat(load_b["src"]).split("[")[0]
    y_name = _flat(store["dst"]).split("[")[0]
    tensors = doc["tensors"]
    for name in (x_name, w_name, y_name):
        _expect(tensors.get(name, {}).get("dtype") == "f16", f"/tensors/{name}", "expected f16")
    _expect(tensors[y_name].get("out") is True, f"/tensors/{y_name}/out", "the store target")
    m, k = (str(d) for d in tensors[x_name]["shape"])
    n, k_w = (str(d) for d in tensors[w_name]["shape"])
    _expect(k_w == k, f"/tensors/{w_name}/shape", f"expected [N, {k}]")
    _expect([str(d) for d in tensors[y_name]["shape"]] == [m, n], f"/tensors/{y_name}/shape", "")
    _expect(set(doc["params"]) == {m, n, k}, "/params", f"expected exactly {m}, {n}, {k}")

    # Slices: the template realizes the canonical block-tile slices only.
    canon = {
        (ia, "src"): f"{x_name}[bm*tile.m:bm*tile.m+tile.m,{kt}*tile.k:{kt}*tile.k+tile.k]",
        (ib, "src"): f"{w_name}[bn*tile.n:bn*tile.n+tile.n,{kt}*tile.k:{kt}*tile.k+tile.k]",
    }
    for (i, key), want in canon.items():
        _expect(_flat(loads[i][key]) == want, f"{_LOOP}/{i}/copy/{key}", f"expected {want}")
    want = f"{y_name}[bm*tile.m:bm*tile.m+tile.m,bn*tile.n:bn*tile.n+tile.n]"
    _expect(_flat(store["dst"]) == want, "/body/3/copy/dst", f"expected {want}")

    # Buffers.
    buffers = doc["buffers"]

    def buffer(name: str, tier: str, dtype: str, shape: tuple[int, int]) -> dict:
        buf, where = buffers[name], f"/buffers/{name}"
        _expect(buf.get("tier") == tier and buf.get("dtype") == dtype, where, f"{tier} {dtype}")
        got = tuple(const_int(d, tile, f"{where}/shape") for d in buf["shape"])
        _expect(got == shape, f"{where}/shape", f"expected {list(shape)}")
        return buf

    def row_stride(name: str, buf: dict, row: int, multiple: int) -> int:
        stride = buf.get("stride") or [row, 1]
        ld = const_int(stride[0], tile, f"/buffers/{name}/stride/0")
        _expect(stride[1] == 1, f"/buffers/{name}/stride/1", "rows must be contiguous")
        _expect(
            ld >= row and ld % multiple == 0,
            f"/buffers/{name}/stride/0",
            f"row stride must be >= {row} and a multiple of {multiple}",
        )
        return ld

    a = buffer(a_buf, "shared", "f16", (bm, bk))
    b = buffer(b_buf, "shared", "f16", (bn, bk))
    stages = const_int(a.get("stages", 1), tile, f"/buffers/{a_buf}/stages")
    b_stages = const_int(b.get("stages", 1), tile, f"/buffers/{b_buf}/stages")
    _expect(b_stages == stages >= 1, f"/buffers/{b_buf}/stages", f"must equal {a_buf}.stages")
    buffer(acc, "register", "f32", (bm, bn))
    c_buf = str(to_shared["dst"])
    c = buffer(c_buf, "shared", "f32", (bm, bn))
    _expect(to_shared.get("src") == acc, "/body/2/copy/src", f"expected {acc}")
    _expect(_flat(store["src"]) == c_buf, "/body/3/copy/src", f"expected {c_buf}")
    _expect(store.get("cast") == "f16", "/body/3/copy/cast", "expected f16")
    _expect(fill.get("dst") == acc, "/body/0/fill/dst", f"expected {acc}")

    # Placement: warp grid from warp_tile.mma; MMA shape, chunk width and ldm rules from target.
    place = doc.get("placement", {}).get(target.name)
    where = f"/placement/{target.name}"
    _expect(place, where, "required until DeviceSpec defaults land")
    _expect(not place.get("specialize"), f"{where}/specialize", "role specialization unsupported")
    wm, wn = (const_int(e, tile, f"{where}/warp_tile/mma") for e in place["warp_tile"]["mma"])
    mma_m, mma_n, mma_k = target.mma
    chunk = target.chunk_bytes // 2
    _expect(bm % wm == 0 and bn % wn == 0, f"{where}/warp_tile/mma", "must divide the tile")
    _expect(
        wm % mma_m == 0 and wn % mma_n == 0,
        f"{where}/warp_tile/mma",
        f"must be a multiple of the {mma_m}x{mma_n} MMA shape",
    )
    _expect(bk % mma_k == 0 and bk % chunk == 0, "/tile/k", f"must be a multiple of {mma_k}")
    warps = (bm // wm, bn // wn)
    _expect(warps[0] * warps[1] == place.get("warps"), f"{where}/warps", f"expected {warps}")

    names = {m: "M", n: "N", k: "K", "tile.m": "BM", "tile.n": "BN", "tile.k": "BK"}
    region = GemmRegion(
        kernel=kernel,
        tile=(bm, bn, bk),
        stages=stages,
        lda=row_stride(a_buf, a, bk, max(target.ldm_multiple_f16, chunk)),
        ldb=row_stride(b_buf, b, bk, max(target.ldm_multiple_f16, chunk)),
        ldc=row_stride(c_buf, c, bn, target.ldm_multiple_f32),
        warps=warps,
        fill=float(fill.get("value", 0.0)),
        async_copy=(load_a.get("async") is True, load_b.get("async") is True),
        pad=(load_a.get("pad") == "zero", load_b.get("pad") == "zero"),
        mask=store.get("mask") is True,
        full_edge=bool(_slots(mm, "wait") & {f"{a_buf}.full", f"{b_buf}.full"}),
        free_edge=f"{a_buf}.free" in _slots(load_a, "wait")
        or f"{b_buf}.free" in _slots(load_b, "wait"),
        out_edge=f"{c_buf}.full" in _slots(store, "wait"),
        trips=to_cpp(loop["trips"], names, "/body/1/for/trips"),
        grid=tuple(to_cpp(doc["grid"][g], names, f"/grid/{g}") for g in ("bm", "bn")),
        assume=tuple(
            (to_cpp(p, names, f"/assume/{i}"), str(p)) for i, p in enumerate(doc.get("assume", []))
        ),
    )
    _expect(set(doc["grid"]) == {"bm", "bn"}, "/grid", "expected bm and bn")
    _expect(
        region.threads <= target.max_threads_per_block,
        f"{where}/warps",
        "too many threads",
        "L-RESOURCE",
    )
    _expect(
        region.shared_bytes <= target.static_shared_bytes,
        "/buffers",
        f"{region.shared_bytes} B of shared memory exceeds {target.static_shared_bytes} B",
        "L-RESOURCE",
    )
    return region


@dataclass(frozen=True)
class LoweredKernel:
    name: str  # extension module name, unique per generated source
    function: str  # host entry point: the PKIR `kernel`
    source: str  # CUDA C++
    region: GemmRegion
    cuda_cflags: tuple[str, ...]

    def build(self):
        """Compile with nvcc and import; needs a CUDA toolchain, so tests call it on the GPU box."""
        from torch.utils.cpp_extension import load_inline

        return load_inline(
            name=self.name,
            cpp_sources=[f"torch::Tensor {self.function}(torch::Tensor x, torch::Tensor w);"],
            cuda_sources=[self.source],
            functions=[self.function],
            extra_cuda_cflags=list(self.cuda_cflags),
            verbose=False,
        )

    @property
    def device_source(self) -> str:
        """The anonymous-namespace device code only, suitable for `nvcc -ptx`.

        The extension source includes torch's host wrapper; that translation unit cannot be
        compiled to PTX on a machine without libtorch. The kernel and its helpers are device-only,
        so strip the host half and supply the two CUDA headers the template uses.
        """
        start = self.source.index("namespace {")
        end = self.source.index("}  // namespace") + len("}  // namespace")
        return (
            "#include <cuda_runtime.h>\n"
            "#include <cuda_fp16.h>\n"
            "#include <mma.h>\n"
            "using namespace nvcuda;\n\n" + self.source[start:end] + "\n"
        )


def lower(doc: Mapping, target: CudaTarget = SM86) -> LoweredKernel:
    try:
        region = match(doc, target)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise LoweringError(
            "L-TEMPLATE", "", f"not a tiled-GEMM region ({type(exc).__name__}: {exc})"
        ) from exc
    source = emit(region, target)
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    flags = ("-O3", f"-gencode=arch=compute_{target.arch},code=sm_{target.arch}")
    return LoweredKernel(f"nomad_{region.kernel}_{digest}", region.kernel, source, region, flags)


def _load(buf: str, ld: str, src: str, rows: str, row0: str, async_copy: bool, pad: bool) -> str:
    """One 16-byte chunk of an operand copy: global row `row0 + r`, columns `k0 + col` onward."""
    gmem = f"{src} + static_cast<size_t>({row0} + r) * K + k0 + col"
    smem = f"{buf} + r * {ld} + col"
    guard = f"const bool ok = {row0} + r < {rows} && k0 + col < K;  // pad: zero\n    "
    if async_copy and pad:
        return f"{guard}cp_async_16({smem}, ok ? {gmem} : {src}, ok);"
    if async_copy:
        return f"cp_async_16({smem}, {gmem}, true);  // no pad declared: unguarded"
    load = f"*reinterpret_cast<const uint4*>({gmem})"
    if pad:
        return f"{guard}*reinterpret_cast<uint4*>({smem}) = ok ? {load} : make_uint4(0, 0, 0, 0);"
    return f"*reinterpret_cast<uint4*>({smem}) = {load};  // no pad declared: unguarded"


def emit(r: GemmRegion, target: CudaTarget = SM86) -> str:
    any_async = any(r.async_copy)
    full = "__syncthreads();  // operand `full`: stage s is visible to every warp"
    if any_async:
        full = (
            f"cp_async_wait<STAGES - 1>();  // this thread's K-tile kt copies landed...\n    {full}"
        )
    store = "y[static_cast<size_t>(m0 + r) * N + n0 + c] = __float2half(Cs[r][c]);"
    checks = "".join(
        f'  TORCH_CHECK({cpp}, "assume violated: {text}");\n' for cpp, text in r.assume
    )
    return _CU.substitute(
        kernel=r.kernel,
        target=target.name,
        bm=r.tile[0],
        bn=r.tile[1],
        bk=r.tile[2],
        mma_m=target.mma[0],
        mma_n=target.mma[1],
        mma_k=target.mma[2],
        warps_m=r.warps[0],
        warps_n=r.warps[1],
        warp_size=target.threads_per_warp,
        stages=r.stages,
        lda=r.lda,
        ldb=r.ldb,
        ldc=r.ldc,
        chunk=target.chunk_bytes // 2,
        load_a=_load("As_st", "LDA", "X", "M", "m0", r.async_copy[0], r.pad[0]),
        load_b=_load("Bs_st", "LDB", "W", "N", "n0", r.async_copy[1], r.pad[1]),
        fill=f"{r.fill!r}f",
        trips=r.trips,
        commit="cp_async_commit();" if any_async else "// no async copies: nothing to commit",
        full_wait=full if r.full_edge else "// the matmul declares no `full` wait: none emitted",
        free_wait="__syncthreads();  // operand `free`: the next prefetch may overwrite stage s"
        if r.free_edge
        else "// the operand copies declare no `free` wait: none emitted",
        out_wait="__syncthreads();  // epilogue `full`: every warp's fragments are in Cs"
        if r.out_edge
        else "// the store declares no `full` wait: none emitted",
        store=f"if (m0 + r < M && n0 + c < N) {store}" if r.mask else store,
        assume_checks=checks,
        grid_m=r.grid[0],
        grid_n=r.grid[1],
    )


_CU = Template(r"""// PKIR kernel `$kernel` lowered for $target by nomad.lowering.cuda.
// Do not edit: change the PKIR document and lower again.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <climits>
#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int BM = $bm, BN = $bn, BK = $bk;                  // tile
constexpr int MMA_M = $mma_m, MMA_N = $mma_n, MMA_K = $mma_k;  // capabilities.tensor_core_mma
constexpr int WARPS_M = $warps_m, WARPS_N = $warps_n;          // placement.$target.warp_tile
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
constexpr int FM = WM / MMA_M, FN = WN / MMA_N;
constexpr int THREADS = WARPS_M * WARPS_N * $warp_size;
constexpr int STAGES = $stages;
constexpr int LDA = $lda, LDB = $ldb, LDC = $ldc;  // buffer row strides, padding kept (spec §8)
constexpr int CHUNK = $chunk;                      // halves per 16-byte copy chunk
constexpr int CHUNKS_PER_ROW = BK / CHUNK;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int src_bytes = pred ? 16 : 0;  // 0 bytes read => the 16-byte destination is zero-filled
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(dst), "l"(gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// The two operand copies of one K-tile into one stage.
__device__ __forceinline__ void load_stage(half* As_st, half* Bs_st, const half* __restrict__ X,
                                           const half* __restrict__ W, int M, int N, int K,
                                           int m0, int n0, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < (BM * CHUNKS_PER_ROW + THREADS - 1) / THREADS; ++i) {
    const int c = tid + i * THREADS;
    if (c >= BM * CHUNKS_PER_ROW) break;
    const int r = c / CHUNKS_PER_ROW, col = (c % CHUNKS_PER_ROW) * CHUNK;
    $load_a
  }
#pragma unroll
  for (int i = 0; i < (BN * CHUNKS_PER_ROW + THREADS - 1) / THREADS; ++i) {
    const int c = tid + i * THREADS;
    if (c >= BN * CHUNKS_PER_ROW) break;
    const int r = c / CHUNKS_PER_ROW, col = (c % CHUNKS_PER_ROW) * CHUNK;
    $load_b
  }
}

__global__ void __launch_bounds__(THREADS)
gemm_kernel(const half* __restrict__ X, const half* __restrict__ W, half* __restrict__ y,
            int M, int N, int K) {
  __shared__ __align__(128) half As[STAGES][BM][LDA];
  __shared__ __align__(128) half Bs[STAGES][BN][LDB];
  __shared__ __align__(128) float Cs[BM][LDC];

  const int tid = threadIdx.x;
  const int warp = tid / $warp_size;
  const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;  // grid.bm on y, grid.bn on x
  const int k_tiles = $trips;

  wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], $fill);

  // Software pipeline (spec §8): K-tiles 0 .. STAGES-2 are issued before the loop, and iteration
  // kt issues K-tile kt + STAGES - 1 into the stage that iteration kt - 1 freed.
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < k_tiles) load_stage(&As[s][0][0], &Bs[s][0][0], X, W, M, N, K, m0, n0, s * BK, tid);
    $commit
  }

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int s = kt % STAGES;
    const int kn = kt + STAGES - 1;
    if (kn < k_tiles) {
      load_stage(&As[kn % STAGES][0][0], &Bs[kn % STAGES][0][0], X, W, M, N, K, m0, n0, kn * BK,
                 tid);
    }
    $commit
    $full_wait

#pragma unroll
    for (int kk = 0; kk < BK; kk += MMA_K) {
      wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> a[FM];
      wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> b[FN];
#pragma unroll
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &As[s][warp_m * WM + i * MMA_M][kk], LDA);
#pragma unroll
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &Bs[s][warp_n * WN + j * MMA_N][kk], LDB);
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    $free_wait
  }

#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j)
      wmma::store_matrix_sync(&Cs[warp_m * WM + i * MMA_M][warp_n * WN + j * MMA_N], acc[i][j],
                              LDC, wmma::mem_row_major);
  $out_wait
  for (int idx = tid; idx < BM * BN; idx += THREADS) {
    const int r = idx / BN, c = idx % BN;
    $store
  }
}

}  // namespace

torch::Tensor $kernel(torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kHalf && w.scalar_type() == torch::kHalf,
              "x and w must be float16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "expected x[M,K], w[N,K]");
  x = x.contiguous();
  w = w.contiguous();
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
$assume_checks  TORCH_CHECK(M <= INT_MAX && N <= INT_MAX && K <= INT_MAX, "sizes exceed int32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "x and w must be 16-byte aligned");
  auto y = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) return y;
  const dim3 grid(static_cast<unsigned>($grid_n), static_cast<unsigned>($grid_m));
  gemm_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()), static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
""")
