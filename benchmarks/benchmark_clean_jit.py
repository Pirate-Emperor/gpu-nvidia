#!/usr/bin/env python3
"""Measure first cuTile launch with fresh cuTile and CUDA driver disk caches.

Each specialization runs in a new Python process with new, empty cache
directories. CUDA context creation and host/GPU input preparation happen before
the timer. The first-call number therefore measures specialization compilation
plus the first launch and synchronization, not Python interpreter startup or a
pure compiler-only duration.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from cases import MANDATORY_CASES


def cache_inventory(path: Path) -> dict[str, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
    }


def parse_case_tile(value: str) -> tuple[str, int, int]:
    try:
        case_name, tile = value.split(":", 1)
        tile_q, tile_k = (int(x) for x in tile.lower().split("x", 1))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "expected CASE:TILE_QxTILE_K, for example long_d128:64x64"
        ) from exc
    known = {case.name for case in MANDATORY_CASES}
    if case_name not in known or tile_q <= 0 or tile_k <= 0:
        raise argparse.ArgumentTypeError(f"invalid case/tile: {value}")
    return case_name, tile_q, tile_k


def worker(args: argparse.Namespace) -> int:
    import cupy as cp
    import cuda.tile  # noqa: F401
    from cutile_attention import launch_device
    from reference import causal_attention_fp32, passes_mixed_tolerance

    case = next(c for c in MANDATORY_CASES if c.name == args.case)
    if case.N % args.tile_q or case.N % args.tile_k:
        raise ValueError(
            f"{case.name}: N={case.N} is not divisible by "
            f"{args.tile_q}x{args.tile_k}"
        )

    rng = np.random.default_rng(case.seed)
    q_np = rng.standard_normal(
        (case.B, case.H, case.N, case.D), dtype=np.float32
    ).astype(np.float16)
    k_np = rng.standard_normal(
        (case.B, case.H, case.N, case.D), dtype=np.float32
    ).astype(np.float16)
    v_np = rng.standard_normal(
        (case.B, case.H, case.N, case.Dv), dtype=np.float32
    ).astype(np.float16)

    # Create the CUDA context and inputs before starting the first-call timer.
    cp.cuda.Device(0).use()
    context_probe = cp.asarray(np.zeros((1,), dtype=np.float32))
    cp.cuda.get_current_stream().synchronize()
    del context_probe
    q, k, v = cp.asarray(q_np), cp.asarray(k_np), cp.asarray(v_np)
    expected = causal_attention_fp32(q_np, k_np, v_np, output_dtype=np.float16)
    stream = cp.cuda.get_current_stream()

    cache_dirs = {
        "cutile": Path(os.environ["CUDA_TILE_CACHE_DIR"]),
        "cuda_driver": Path(os.environ["CUDA_CACHE_PATH"]),
        "cutile_temp": Path(os.environ["CUDA_TILE_TEMP_DIR"]),
        "cupy": Path(os.environ["CUPY_CACHE_DIR"]),
    }
    before = {name: cache_inventory(path) for name, path in cache_dirs.items()}
    if before["cutile"]["file_count"] or before["cuda_driver"]["file_count"]:
        raise RuntimeError(
            "Clean-cache precondition failed; cuTile/driver cache was not empty: "
            + json.dumps(before, sort_keys=True)
        )

    def launch():
        return launch_device(
            q, k, v, tile_q=args.tile_q, tile_k=args.tile_k
        )

    started = time.perf_counter()
    output = launch()
    stream.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1000.0
    actual = cp.asnumpy(output)
    correctness_pass = (
        actual.shape == expected.shape
        and actual.dtype == np.float16
        and np.isfinite(actual).all()
        and passes_mixed_tolerance(
            actual, expected, abs_scale=2e-2, rel_scale=2e-2
        )
    )
    if not correctness_pass:
        raise AssertionError(
            f"{case.name}/{args.tile_q}x{args.tile_k}: first output failed NumPy oracle"
        )

    # Steady-state kernel latency is measured after the first specialization
    # compile. CUDA events isolate device execution from host wall-clock time.
    for _ in range(args.warmup):
        launch()
    stream.synchronize()
    samples_ms: list[float] = []
    for _ in range(args.iterations):
        start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
        start_event.record(stream)
        for _ in range(args.repeats_per_sample):
            launch()
        end_event.record(stream)
        end_event.synchronize()
        samples_ms.append(
            float(cp.cuda.get_elapsed_time(start_event, end_event))
            / args.repeats_per_sample
        )

    after = {name: cache_inventory(path) for name, path in cache_dirs.items()}
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props.get("name", b"<unknown>")
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")
    result = {
        "case": case.name,
        "B": case.B, "H": case.H, "N": case.N,
        "D": case.D, "Dv": case.Dv,
        "tile_q": args.tile_q, "tile_k": args.tile_k,
        "gpu": str(gpu_name),
        "compute_capability": f'{props.get("major", "?")}.{props.get("minor", "?")}',
        "cuda_driver_api_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_api_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_tile": importlib.metadata.version("cuda-tile"),
        "git_commit": os.environ.get("GITHUB_SHA"),
        "first_call_wall_ms": first_call_ms,
        "first_call_scope": (
            "Fresh Python process and initially empty CUDA_TILE_CACHE_DIR and "
            "CUDA_CACHE_PATH; CUDA context, imports and input preparation are "
            "excluded; includes cuTile specialization compile, first kernel "
            "launch and device synchronization."
        ),
        "steady_state": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats_per_sample": args.repeats_per_sample,
            "median_ms": float(np.median(samples_ms)),
            "p90_ms": float(np.percentile(samples_ms, 90)),
            "samples_ms": samples_ms,
        },
        "correctness_vs_numpy": correctness_pass,
        "cache_before": before,
        "cache_after": after,
    }
    Path(args.worker_output).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("CLEAN_JIT_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/cutile-clean-jit.json"))
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case_tile,
        default=[
            ("long_d128", 64, 64),
            ("longer_batched_d128", 64, 128),
        ],
        metavar="CASE:TQxTK",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats-per-sample", type=int, default=5)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--tile-q", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--tile-k", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        return worker(args)
    if args.warmup < 1 or args.iterations < 30 or args.repeats_per_sample < 1:
        parser.error("warmup >= 1, iterations >= 30 and repeats-per-sample >= 1 required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_root = args.output.parent / "clean-jit-isolated-caches"
    cache_root.mkdir(parents=True, exist_ok=True)
    records = []
    for case_name, tile_q, tile_k in args.cases:
        label = f"{case_name}-tq{tile_q}-tk{tile_k}"
        cache_home = Path(tempfile.mkdtemp(prefix=label + "-", dir=cache_root))
        dirs = {
            "CUDA_TILE_CACHE_DIR": cache_home / "cutile",
            "CUDA_TILE_TEMP_DIR": cache_home / "tile-temp",
            "CUDA_CACHE_PATH": cache_home / "cuda-driver",
            "CUPY_CACHE_DIR": cache_home / "cupy",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        worker_path = cache_home / "worker-result.json"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--_worker", "--case", case_name,
            "--tile-q", str(tile_q), "--tile-k", str(tile_k),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--repeats-per-sample", str(args.repeats_per_sample),
            "--worker-output", str(worker_path),
        ]
        env = os.environ.copy()
        env.update({key: str(path) for key, path in dirs.items()})
        env["CUDA_CACHE_DISABLE"] = "0"
        print(f"START_CLEAN_CACHE_SPECIALIZATION={label}", flush=True)
        proc = subprocess.run(
            command, cwd=ROOT, env=env, text=True,
            capture_output=True, check=False,
        )
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.returncode != 0:
            records.append({
                "case": case_name, "tile_q": tile_q, "tile_k": tile_k,
                "status": "FAIL", "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-8000:],
                "cache_home": str(cache_home),
            })
            raise RuntimeError(
                f"clean-cache specialization failed: {label}; see captured stderr"
            )
        if not worker_path.is_file():
            raise RuntimeError(f"worker result missing for {label}")
        result = json.loads(worker_path.read_text(encoding="utf-8"))
        result["cache_home"] = str(cache_home)
        result["status"] = "PASS"
        records.append(result)
        print(
            f"{label}: clean_first_call={result['first_call_wall_ms']:.3f} ms; "
            f"steady_median={result['steady_state']['median_ms']:.6f} ms"
        )

    payload = {
        "title": "Fresh-cache cuTile JIT/first-call and steady-state measurements",
        "method": (
            "One new Python process and initially empty cuTile and CUDA driver "
            "disk-cache directories per specialization. CUDA context initialization, "
            "imports and input transfer are outside the first-call wall timer. "
            "First call includes cuTile specialization, launch and synchronization; "
            "steady-state latency is measured separately with CUDA events."
        ),
        "settings": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats_per_sample": args.repeats_per_sample,
        },
        "results": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"CLEAN_JIT_REPORT={args.output}")
    print(f"CLEAN_JIT_CASES={len(records)}")
    print(f"CLEAN_JIT_ALL_PASSED={all(x.get('status') == 'PASS' for x in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
