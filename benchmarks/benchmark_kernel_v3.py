#!/usr/bin/env python3
"""Benchmark v2 versus experimental v3 under identical resident-input conditions."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests"),str(ROOT/"src"/"variants")]
from cases import MANDATORY_CASES
from shared_tile_v2 import launch_device as v2_launch
from shared_tile_v3 import launch_device as v3_launch
CASES=("short_d64","medium_d64","medium_d128","long_d128","longer_batched_d128")
TILES=(16,32,64)

def make_inputs(case,cp):
    rng=np.random.default_rng(case.seed)
    q=rng.standard_normal((case.B,case.H,case.N,case.D),dtype=np.float32).astype(np.float16)
    k=rng.standard_normal((case.B,case.H,case.N,case.D),dtype=np.float32).astype(np.float16)
    v=rng.standard_normal((case.B,case.H,case.N,case.Dv),dtype=np.float32).astype(np.float16)
    return cp.asarray(q),cp.asarray(k),cp.asarray(v)

def timed(fn,cp,warmup,iters):
    s=cp.cuda.get_current_stream()
    for _ in range(warmup): fn()
    s.synchronize(); a=[]
    for _ in range(iters):
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(s); fn(); e1.record(s); e1.synchronize(); a.append(float(cp.cuda.get_elapsed_time(e0,e1)))
    x=np.asarray(a,dtype=np.float64)
    return float(np.median(x)),float(np.percentile(x,90))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--warmup",type=int,default=10); ap.add_argument("--iterations",type=int,default=30); ap.add_argument("--output",type=Path,default=Path("results/kernel_v3_benchmark.json")); a=ap.parse_args()
    if a.iterations<30 or a.warmup<1: ap.error("warmup>=1 and iterations>=30 required")
    import cupy as cp
    by={c.name:c for c in MANDATORY_CASES}; rows=[]
    for name in CASES:
        case=by[name]; q,k,v=make_inputs(case,cp)
        for ver,fn in (("v2",v2_launch),("v3",v3_launch)):
            for tk in TILES:
                med,p90=timed(lambda f=fn,t=tk:f(q,k,v,tile_k=t),cp,a.warmup,a.iterations)
                rows.append({"case":name,"B":case.B,"H":case.H,"N":case.N,"D":case.D,"tile_k":tk,"version":ver,"warmup":a.warmup,"iterations":a.iterations,"median_ms":med,"p90_ms":p90})
        del q,k,v; cp.get_default_memory_pool().free_all_blocks()
    piv={}
    for r in rows: piv.setdefault((r["case"],r["tile_k"]),{})[r["version"]]=r["median_ms"]
    for r in rows: r["v2_v3_speedup"]=piv[(r["case"],r["tile_k"])]["v2"]/r["median_ms"] if r["version"]=="v3" else piv[(r["case"],r["tile_k"])]["v3"]/r["median_ms"]
    out=a.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"gpu":str(cp.cuda.runtime.getDeviceProperties(0)["name"]),"warmup":a.warmup,"iterations":a.iterations,"cases":list(CASES),"tiles":list(TILES),"results":rows},indent=2)+"\n")
    with out.with_suffix(".csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    for name in CASES:
        for tk in TILES:
            p=piv[(name,tk)]; print(f"{name:24s} tk={tk:2d} v2={p["v2"]:.3f} ms v3={p["v3"]:.3f} ms v2/v3={p["v2"]/p["v3"]:.2f}x")

if __name__=="__main__": raise SystemExit(main())
