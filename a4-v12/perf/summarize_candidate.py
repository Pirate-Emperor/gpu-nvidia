#!/usr/bin/env python3
from __future__ import annotations
import csv, math, random, statistics, sys, os
from collections import defaultdict
from pathlib import Path

src, csvout, mdout = map(Path, sys.argv[1:])
arch = os.environ.get("ARCH", "sm_70")
informational = arch == "sm_80"
rows = list(csv.DictReader(src.open(newline="")))
if not rows:
    raise SystemExit("No paired timing rows found")
by = defaultdict(list)
for r in rows:
    if r["baseline_match"] != "PASS" or r["candidate_match"] != "PASS":
        raise SystemExit("Timed-output correctness status is not PASS")
    base, cand = float(r["baseline_ms"]), float(r["candidate_ms"])
    if base <= 0 or cand <= 0:
        raise SystemExit("Invalid timing sample")
    by[r["case"]].append((base, cand, base/cand))
if len(by) != 13 or any(len(v) < 9 for v in by.values()):
    raise SystemExit("Promotion requires 13 cases and >=9 paired samples per case")

out = []
logs = []
for name, vals in by.items():
    bs, cs, rs = [v[0] for v in vals], [v[1] for v in vals], [v[2] for v in vals]
    bm, cm, mr = statistics.median(bs), statistics.median(cs), statistics.median(rs)
    out.append({"case": name, "base": bm, "cand": cm, "ratio": mr, "pct": (bm/cm-1)*100,
                "bs": statistics.pstdev(bs), "cs": statistics.pstdev(cs)})
    logs.append([math.log(x) for x in rs])
geo = math.exp(sum(sum(v) for v in logs)/sum(map(len, logs)))
random.seed(20261010)
boots = []
for _ in range(10000):
    v = []
    for xs in logs: v.extend(random.choices(xs, k=len(xs)))
    boots.append(math.exp(sum(v)/len(v)))
boots.sort()
lo, hi = boots[int(.025*len(boots))], boots[min(len(boots)-1, int(.975*len(boots)))]
bad = [r for r in out if r["ratio"] < 1/1.10]
ok = geo >= 1.03 and lo > 1.0 and not bad
with csvout.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case","baseline_median_ms","candidate_median_ms","median_paired_speedup","median_improvement_pct","baseline_std_ms","candidate_std_ms"])
    for r in out: w.writerow([r["case"],f'{r["base"]:.6f}',f'{r["cand"]:.6f}',f'{r["ratio"]:.6f}',f'{r["pct"]:.2f}',f'{r["bs"]:.6f}',f'{r["cs"]:.6f}'])
with mdout.open("w") as f:
    f.write("# Candidate performance comparison\n\n")
    f.write(f"- Compile target: {arch}\n")
    warmups = int(os.environ.get("WARM", "1"))
    f.write(f"- Untimed per-case warm-ups: {warmups} per source, immediately before timed samples; excluded from timing data.\n")
    if informational: f.write("- Informational target control only; it does not determine source promotion.\n")
    f.write(f"- Cases: {len(out)}/13; paired timing samples per case: {len(rows)//len(out)}\n")
    f.write("- Both sources passed golden tests; candidate passed repeats; every timed output was checked again.\n")
    f.write(f"- Geometric mean paired speedup (baseline/candidate): **{geo:.4f}x**\n")
    f.write(f"- Stratified bootstrap 95% interval: **[{lo:.4f}x, {hi:.4f}x]**\n")
    f.write(f"- Worst per-case median paired speedup: **{min(r['ratio'] for r in out):.4f}x**\n")
    f.write(f"- Cases >10% slower: {', '.join(r['case'] for r in bad) or 'none'}\n")
    gate_label = "Informational diagnostic gate (not used for promotion)" if informational else "Promotion gate"
    f.write(f"- {gate_label}: **{'PASS' if ok else 'FAIL / INCONCLUSIVE'}** (requires >=1.03x geomean, interval lower bound >1, no >10% per-case regression).\n\n")
    f.write("| Case | Baseline median (ms) | Candidate median (ms) | Paired median speedup | Improvement | Baseline σ | Candidate σ |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in out: f.write(f"| {r['case']} | {r['base']:.3f} | {r['cand']:.3f} | {r['ratio']:.3f}x | {r['pct']:.1f}% | {r['bs']:.3f} | {r['cs']:.3f} |\n")
print(mdout.read_text())
if not ok:
    print("PERFORMANCE_PROMOTION_GATE=FAIL" if not informational else "PERFORMANCE_DIAGNOSTIC_ONLY=FAIL")
    if not informational: raise SystemExit(1)
else:
    print("PERFORMANCE_PROMOTION_GATE=PASS" if not informational else "PERFORMANCE_DIAGNOSTIC_ONLY=PASS")
