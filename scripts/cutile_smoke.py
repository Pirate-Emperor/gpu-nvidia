#!/usr/bin/env python
"""Compile, launch, and verify a tiny genuine NVIDIA cuTile kernel.

This is an environment smoke test, not a surrogate CUDA kernel and not an
attention correctness/performance result. It makes no system changes.
"""

from __future__ import annotations

from importlib import metadata
import platform
import sys

import numpy as np


def distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<not installed as a distribution>"


def main() -> int:
    print("CUTILE_SMOKE=START")
    print(f"OS={platform.platform()}")
    print(f"ARCH={platform.machine()}")
    print(f"PYTHON={sys.version.split()[0]}")
    for package in (
        "numpy",
        "cupy-cuda13x",
        "cuda-tile",
        "nvidia-cuda-tileiras",
        "nvidia-cuda-nvcc",
        "nvidia-nvvm",
    ):
        print(f"PACKAGE_{package.upper().replace('-', '_')}={distribution_version(package)}")

    # Import only after printing the environment so missing runtime pieces are
    # distinguishable from a compiler/launch failure in the captured log.
    import cupy as cp
    import cuda.tile as ct

    device_id = cp.cuda.Device().id
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    name = props.get("name", b"<unknown>")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    major = int(props.get("major", -1))
    minor = int(props.get("minor", -1))
    print(f"GPU={name}")
    print(f"GPU_COMPUTE_CAPABILITY={major}.{minor}")
    print(f"CUDA_DRIVER_API_VERSION={cp.cuda.runtime.driverGetVersion()}")
    print(f"CUDA_RUNTIME_API_VERSION={cp.cuda.runtime.runtimeGetVersion()}")
    print(f"CUPY_VERSION={cp.__version__}")
    print(f"CUTILE_MODULE={getattr(ct, '__file__', '<unknown>')}")

    @ct.kernel
    def vector_add(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
        b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
        ct.store(c, index=(pid,), tile=a_tile + b_tile)

    n = 4096
    tile_size = 256
    host_a = np.arange(n, dtype=np.float32)
    host_b = np.full(n, np.float32(3.25), dtype=np.float32)
    a = cp.asarray(host_a)
    b = cp.asarray(host_b)
    c = cp.empty_like(a)

    # ct.launch compiles the actual TileIR kernel if no matching cache entry
    # exists, then launches it on the visible NVIDIA GPU.
    ct.launch(
        cp.cuda.get_current_stream(),
        (ct.cdiv(n, tile_size), 1, 1),
        vector_add,
        (a, b, c, tile_size),
    )
    cp.cuda.get_current_stream().synchronize()
    actual = cp.asnumpy(c)
    expected = host_a + host_b
    np.testing.assert_array_equal(actual, expected)

    print("KERNEL=vector_add")
    print(f"ELEMENTS={n}")
    print(f"TILE_SIZE={tile_size}")
    print("GPU_SYNCHRONIZE=PASS")
    print("OUTPUT_CHECK=PASS")
    print("CUTILE_SMOKE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
