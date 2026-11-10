#!/usr/bin/env python3
"""Create normalized metric and instruction-count comparisons from NCU exports."""
import argparse, csv, json, re
from pathlib import Path

METRICS = [
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__block_size", "launch__grid_size",
    "launch__registers_per_thread", "launch__registers_per_thread_allocated",
    "launch__shared_mem_per_block_static", "launch__shared_mem_per_block",
    "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_registers",
    "launch__occupancy_limit_warps", "launch__waves_per_multiprocessor",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "memory_l1_wavefronts_shared", "memory_l1_wavefronts_shared_ideal",
    "smsp__inst_executed.sum",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
]

def read_raw(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise ValueError("Expected three raw-export rows: names, units, values")
    return {n: (u.strip(), v.strip()) for n, u, v in zip(rows[0], rows[1], rows[2])}

def num(v):
    try:
        return float(v.replace(",", "")) if v not in ("", "-", None) else None
    except ValueError:
        return None

def fmt(v):
    if v is None: return "—"
    if abs(v) >= 100000: return f"{v:,.0f}"
    if abs(v) >= 1000: return f"{v:,.1f}"
    return f"{v:.4f}".rstrip("0").rstrip(".")

def read_details(path):
    t = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out = {}
    labels = [
        "No Eligible", "One or More Eligible", "Active Warps Per Scheduler",
        "Eligible Warps Per Scheduler", "Issued Warp Per Scheduler",
        "Warp Cycles Per Issued Instruction", "Warp Cycles Per Executed Instruction",
        "Memory Throughput", "Mem Busy", "Max Bandwidth", "L2 Hit Rate", "Mem Pipes Busy",
        "Compute (SM) Throughput", "Duration", "SM Active Cycles", "Executed Instructions",
    ]
    for label in labels:
        m = re.search(rf"(?m)^\s*{re.escape(label)}\s+[^\n]*?\s([0-9][0-9,]*(?:\.[0-9]+)?)\s*$", t)
        if m: out[label] = num(m.group(1))
    for key, pattern in {
        "Math Pipe Throttle Stall Cycles": r"spends\s+([0-9.]+)\s+cycles being stalled waiting for the execution pipe",
        "Memory Throughput GB/s": r"Memory Throughput\s+Gbyte/s\s+([0-9][0-9,]*(?:\.[0-9]+)?)",
    }.items():
        m = re.search(pattern, t, re.I)
        if m: out[key] = num(m.group(1))
    m = re.search(r"Est\. Speedup:\s*([0-9.]+)%\s+This workload executes\s+([0-9,]+)\s+fused and\s+([0-9,]+)\s+non-fused FP32 instructions", t, re.S)
    if m:
        out["FP32 Non-Fused Overall Estimated Speedup"] = num(m.group(1))
        out["FP32 Fused Instructions"] = num(m.group(2))
        out["FP32 Non-Fused Instructions"] = num(m.group(3))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_a = read_raw(data/"medium_d128_64x64-raw.csv")
    raw_b = read_raw(data/"medium_d128_64x128-raw.csv")
    rows = []
    for m in METRICS:
        if m not in raw_a and m not in raw_b: continue
        unit = (raw_a.get(m) or raw_b.get(m))[0]
        va = num(raw_a.get(m, (unit, ""))[1])
        vb = num(raw_b.get(m, (unit, ""))[1])
        delta = vb-va if va is not None and vb is not None else None
        rel = delta/va*100 if delta is not None and va not in (None, 0) else None
        rows.append({"metric":m,"unit":unit,"64x64":fmt(va),"64x128":fmt(vb),
                     "delta_64x128_minus_64x64":fmt(delta),"relative_change_pct":fmt(rel)})
    da = read_details(data/"medium_d128_64x64-details.txt")
    db = read_details(data/"medium_d128_64x128-details.txt")
    detail_units = {
        "No Eligible":"%", "One or More Eligible":"%",
        "Active Warps Per Scheduler":"warp", "Eligible Warps Per Scheduler":"warp",
        "Issued Warp Per Scheduler":"warp", "Warp Cycles Per Issued Instruction":"cycle",
        "Warp Cycles Per Executed Instruction":"cycle", "Memory Throughput":"%",
        "Mem Busy":"%", "Max Bandwidth":"%", "L2 Hit Rate":"%", "Mem Pipes Busy":"%",
        "Compute (SM) Throughput":"%", "Duration":"us", "SM Active Cycles":"cycle",
        "Executed Instructions":"inst", "Math Pipe Throttle Stall Cycles":"cycle",
        "Memory Throughput GB/s":"Gbyte/s",
        "FP32 Non-Fused Overall Estimated Speedup":"% (NCU local estimate)",
        "FP32 Fused Instructions":"inst", "FP32 Non-Fused Instructions":"inst",
    }
    for k, unit in detail_units.items():
        va, vb = da.get(k), db.get(k)
        delta = vb-va if va is not None and vb is not None else None
        rel = delta/va*100 if delta is not None and va not in (None, 0) else None
        rows.append({"metric":"details:"+k,"unit":unit,"64x64":fmt(va),"64x128":fmt(vb),
                     "delta_64x128_minus_64x64":fmt(delta),"relative_change_pct":fmt(rel)})
    ca = num(raw_a.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",("", ""))[1]) or 0
    cb = num(raw_b.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",("", ""))[1]) or 0
    wa = num(raw_a.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",("", ""))[1]) or 0
    wb = num(raw_b.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",("", ""))[1]) or 0
    pa, pb = (ca/wa*100 if wa else 0), (cb/wb*100 if wb else 0)
    rows.append({"metric":"derived_shared_bank_conflicts_per_shared_wavefront_pct","unit":"%",
                 "64x64":fmt(pa),"64x128":fmt(pb),"delta_64x128_minus_64x64":fmt(pb-pa),"relative_change_pct":"—"})
    with (out/"metric-comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    inst_rows=[]
    for cfg in ("64x64","64x128"):
        sass=(data/f"medium_d128_{cfg}-source-tileir-sass.txt").read_text(encoding="utf-8-sig",errors="replace")
        pp=data/f"medium_d128_{cfg}-source-tileir-ptx.txt"
        ptx=pp.read_text(encoding="utf-8-sig",errors="replace") if pp.exists() else ""
        for view, text, ops in [
            ("SASS",sass,["HMMA.16816.F32","LDGSTS.E.128","LDSM.16.M88.4","BAR.SYNC.DEFER_BLOCKING","DEPBAR.LE"]),
            ("PTX",ptx,["cp.async.ca.shared.global","cp.async.commit_group","cp.async.wait_group","ldmatrix.sync.aligned.m8n8.x4.shared.b16","bar.sync","ex2.approx.ftz.f32","fma.rn.f32","mul.rn.f32"]),
        ]:
            for op in ops:
                inst_rows.append({"config":cfg,"view":view,"instruction":op,
                                  "static_listing_occurrences":text.count(op),
                                  "note":"Static listing count only; not a dynamic executed-instruction count."})
    with (out/"source-instruction-counts.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(inst_rows[0])); w.writeheader(); w.writerows(inst_rows)

    md = [
        "# Metric analysis and interpretation", "",
        "## How to read this comparison", "",
        "The raw CSV is a three-row export: metric names, units, then values. metric-comparison.csv normalizes selected values from the 1,802-column raw exports and adds Details-page metrics not represented as raw counters.",
        "Nsight Compute replay/profiling duration is instrumented timing. Do not use 52.86 vs 54.50 us as the final tile winner; confirm with repeated unprofiled CUDA-event timing.", "",
        "## Primary findings", "",
        "- Shared memory limits occupancy: one block/SM in both configurations, while the register limit allows two. Static shared memory rises from 65.536 to 81.920 KB/block and occupancy stays about 8.33%.",
        "- Scheduler starvation is severe: about one active warp/scheduler, 0.14 eligible warps/scheduler and 85.90–86.14% No Eligible cycles.",
        "- Dominant stalls are math-pipe throttle (2.4/2.6 cycles) and long-scoreboard (about 1.86 in both). Stall ratios are not exclusive percentages and must not be summed.",
        "- Not DRAM-bandwidth saturated: DRAM is about 22–23% of peak, L2 hit rate about 73–75%, and memory throughput about 60.35/58.80 GB/s.",
        "- Shared-bank conflicts are secondary: 64x128 has 5,716 events over 459,417 shared wavefronts (~1.24%); instrumented shared wavefronts equal ideal.",
        "- Local/shared spill requests are zero and branch efficiency is high. Avoid optimizing spilling or divergence that the counters do not show.",
        "- 64x128 reduces K/V loop visits from 36 to 20, but raises registers by 24/thread, shared memory by 16.384 KB/block, static SASS HMMA listing occurrences from 128 to 256, and dynamic instructions by about 3.5%, without increasing occupancy.",
        "", "## PTX, SASS and source interpretation", "",
        "- PTX shows cp.async global-to-shared staging, commit/wait groups, bar.sync and ldmatrix operand loads. Static PTX listing counts are 16/32 cp.async, 32/64 ldmatrix, 34/66 ex2.approx, 104/200 fma.rn and 130/194 mul.rn occurrences for 64x64/64x128; these are static occurrences, not dynamic instruction counts.",
        "- SASS shows LDGSTS, LDSM, BAR.SYNC/DEPBAR and HMMA.16816.F32: the matrix path reaches hardware matrix instructions.",
        "- TileIR sampled stalls concentrate at QK MMA (TileIR line 133) and PV MMA (line 164): 64x64 counts 624/489 all samples (594/460 not-issued); 64x128 counts 663/553 (628/500 not-issued). These are samples, not elapsed-time shares.",
        "- The Source Markers GUI frame is the reliable selected cutile_attention.py source view; the earlier source-high-level-* and source-line91-sass images still show generated _stub.py, not the Python attention source. The PTX GUI images show the PTX header/entry parameters only, not the instruction body. The source-marker left pane shows Instructions Executed attribution: line 87 ct.exp 23.37%/25.65%; line 91 PV MMA 9.82%/9.75%; line 95 output reshape 11.29%/10.90%; line 90 scale multiply 5.19%/2.79%; line 88 softmax sum/update 3.57%/3.31%; line 96 store 1.97%/2.21% (64x64/64x128). These are instruction-attribution values, not runtime shares.",
        "- Source Markers differ by tile: 64x64 flags FP32/64 non-fused instructions at lines 87–88 (softmax exp/update); 64x128 flags high Tensor-pipeline use at QK MMA line 78 and FP32/64 instructions at scale line 79. PTX already contains ex2.approx.ftz.f32 and fma.rn.f32, so blindly adding fast-math/FMA flags may not change the relevant path and could alter numerical behavior. NCU estimates only about 1.1–1.3% overall opportunity; validate accuracy and generated code before considering it.",
        "", "## Recommended experiment order", "",
        "1. Test tile_k=32 with tile_q=64 as a controlled hypothesis. Target shared memory below roughly half the SM shared-memory capacity so two CTAs may become resident. Verify the shared-memory occupancy limit changes from 1 to 2 and eligible warps improve. This may double loop visits, so correctness and unprofiled latency decide.",
        "2. Test tile_q=32 separately. Check registers, shared memory, grid/tail effect and runtime; do not combine changes.",
        "3. Benchmark 64x64 vs 64x128 with warmup and repeated unprofiled CUDA-event timing; report median and spread.",
        "4. Only then consider pipeline depth, instruction fusion or barrier changes. Bank conflicts are low, spills are zero, and FP32 fusion's estimated overall opportunity is small.",
        "", "## Caveats", "",
        "- NCU estimated local speedups overlap and are not additive or guaranteed end-to-end speedups.",
        "- PM sampling is statistical; sample counts are not exact elapsed-time percentages.",
        "- Initial Source captures show generated _stub.py; later source-high-level captures select the actual cutile_attention.py.",
        "- Aggregate shared-bank-conflict counters can include arbitration effects; compare with wavefronts/ideal and stall reasons before attributing runtime cost.",
        "",
    ]
    (out/"metric-analysis.md").write_text("\n".join(md), encoding="utf-8")
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8-sig"))
    notes = [
        "# Screenshot-by-screenshot interpretation", "",
        "Each image below is linked to its exact capture and interpreted against the visible page/panel. Pair these notes with metric-comparison.csv, source-instruction-counts.csv and the full raw/details exports.", ""
    ]
    for capture in manifest.get("captures", []):
        name = Path(capture.get("screenshot", "")).name
        view = capture.get("view", "")
        cfg = "64x128" if "64x128" in name else "64x64"
        no_eligible = "86.14%" if cfg == "64x128" else "85.90%"
        math_stall = "2.6 cycles" if cfg == "64x128" else "2.4 cycles"
        if view == "summary":
            desc = f"Summary for {cfg}: profiled duration, GPU throughput, launch shape, registers/shared memory and NCU local estimates. This is the navigation overview, not a normal-runtime benchmark or guaranteed speedup."
        elif view.startswith("details-"):
            i = int(view.split("-")[1])
            captions = {
                1: "Tile Statistics, GPU Speed of Light and PM Sampling; confirms the 64-CTA grid, 128-thread block and profiling pass context.",
                2: "Speed-of-Light and Memory Workload; low DRAM peak use, L2 hit rate, memory-pipe activity and zero spills argue against a DRAM-bandwidth-first rewrite.",
                3: "Scheduler Statistics begins; it shows the small ready-warp pool relative to active warps.",
                4: f"Scheduler Statistics: No Eligible is {no_eligible}, with about 0.14 eligible warps/scheduler and one active warp/scheduler. This is the strongest scheduler-starvation evidence.",
                5: "Warp State Statistics: about 7.06/7.22 warp cycles per issued instruction, showing sparse issue activity.",
                6: f"Math Pipe Throttle: average stall is {math_stall}. NCU's local-speedup estimate is diagnostic, not a guaranteed end-to-end gain.",
                7: "Instruction Statistics: total executed instructions and spill requests. Zero local/shared spills rules out spilling as the immediate cause.",
                8: "FP32 non-fused instruction summary: fused/non-fused counts and a small overall fusion opportunity; lower priority than occupancy and warp readiness.",
                9: "Combined scheduler, warp-state and instruction view; cross-checks No Eligible, math-pipe stall, instruction count and FP32 non-fused estimates."
            }
            desc = captions[i] + " Use the corresponding rows in metric-comparison.csv for the paired values."
        elif view.startswith("source-high-level-"):
            desc = "Actual cutile_attention.py is selected in the left Source pane, not generated _stub.py. The head/tail captures show different parts of the high-level implementation and per-line instruction attribution. These percentages describe instruction attribution, not runtime."
        elif view.startswith("source-line91-"):
            desc = "Source line 91 (u = ct.mma(p16, v0, u0), the PV MMA) is selected and correlated with the right-hand SASS/PTX pane. This is a concrete source-to-instruction link; inspect HMMA and operand loads plus the lower dependency/statistics tabs."
        elif view.startswith("source-horizontal-"):
            panel = view.replace("source-horizontal-", "")
            panel_names = {
                "left-code": "left Source-code subpanel",
                "left-metrics": "left source-metrics subpanel",
                "right-code": "right SASS-code subpanel",
                "right-metrics": "right SASS-metrics subpanel"
            }
            desc = f"Horizontal-scroll capture of the {panel_names.get(panel, panel)}. The four subpanels are independent; code and metric panes expose different hidden columns. Cite only headers and values actually visible in this image."
        elif view.startswith("source-ptx-horizontal-"):
            desc = "Horizontal-scroll capture of the PTX code or PTX metric subpanel. It exposes columns clipped in the default PTX view; use PTX for the intermediate path and SASS for the executed machine instructions."
        elif view == "source-ptx":
            desc = "PTX is selected in the right Source pane beside the actual Python source. PTX shows asynchronous copies, wait/commit groups, barriers and ldmatrix operand loads; SASS confirms the final machine instructions."
        elif view == "source-view-picker":
            desc = "Right-hand source-view selector immediately before switching from SASS to PTX. The following source-ptx image is the actual PTX evidence."
        elif view == "source-bottom-instructions-dependencies":
            desc = "Lower Instructions & Dependencies tab: Self Instruction Mix, Input Scoreboard Dependencies and Output Scoreboard Dependencies. Empty dependency panes mean no individual instruction was attributed there in that frame, not that dependencies do not exist."
        elif view == "source-bottom-inline-functions":
            desc = "Lower Inline Functions tab. A blank panel means no active inline-function selection was made; it is not proof that the kernel has no inline functions."
        elif view == "source-bottom-source-markers":
            desc = "Lower Source Markers tab. It flags FP32/64 non-fused instructions around source lines 87–88; compare this with the Details-page estimate of only about 1.1–1.3% overall opportunity."
        elif view == "source-bottom-statistics":
            desc = "Lower Statistics tab: live registers, attributed stalls, warp-stall samples, instruction mix, instructions executed, address-space/access details and shared-wavefront metrics for the selected source/instruction."
        elif re.fullmatch(r"source-0[1-5]", view):
            desc = "Initial Source/SASS correlation. The left source is generated _stub.py, not the complete high-level kernel; later source-high-level captures explicitly select cutile_attention.py. The right pane is SASS."
        elif view == "raw":
            desc = "Raw metric table. The full export has far more columns than fit on screen; metric-comparison.csv is the normalized unit-labelled subset and the original raw CSV remains the provenance."
        else:
            desc = "Supporting Nsight Compute view. Interpret only the panel and metric headers actually visible in this frame."
        notes.extend([f"## {name}", "", f"![{name}](screenshots/{name})", "", desc, ""])
    if manifest.get("skipped"):
        notes.extend(["## Skipped captures", ""] + [f"- {x.get('report')}: {x.get('reason')}" for x in manifest["skipped"]] + [""])
    (out / "screenshot-notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Created {len(rows)} metric rows and {len(inst_rows)} static instruction-count rows.")
if __name__ == "__main__":
    main()
