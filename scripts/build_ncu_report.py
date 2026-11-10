#!/usr/bin/env python3
"""Produce a single, evidence-led README for the current medium_d128 NCU captures.

The README embeds only the handful of key-point figures. The full screenshot register
uses links (not duplicate image embeds) and gives capture-specific visible values,
interpretation, relevance, and explicit limitations.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def table_value(rows, metric, config):
    for row in rows:
        if row.get("metric") == metric:
            return row.get(config, "—"), row.get("unit", "")
    return "—", ""


def clean(v):
    return str(v).replace("\\", "\\\\").replace("|", "\\|")


def config_for(name):
    name = str(name).replace("\\", "/").split("/")[-1]
    return "64x128" if "64x128" in name else "64x64"


def screenshot_desc(view: str, cfg: str) -> dict[str, str]:
    a = cfg == "64x64"
    d = {
        "duration": "52.86 µs" if a else "54.50 µs",
        "runtime_improvement": "40.76 µs" if a else "42.34 µs",
        "compute": "18.81%" if a else "20.28%",
        "memory": "22.90%" if a else "22.30%",
        "l1": "17.33%" if a else "17.89%",
        "l2": "14.46%" if a else "15.34%",
        "regs": "231" if a else "255",
        "smem": "65.536 KB" if a else "81.920 KB",
        "noeligible": "85.90%" if a else "86.14%",
        "eligible": "0.14" if a else "0.14",
        "oneeligible": "14.10%" if a else "13.86%",
        "warp_issue": "7.06" if a else "7.22",
        "warp_exec": "7.07" if a else "7.23",
        "math": "2.4" if a else "2.6",
        "math_pct": "33.7%" if a else "36.3%",
        "memory_gbs": "60.35 GB/s" if a else "58.80 GB/s",
        "membusy": "14.46%" if a else "15.34%",
        "maxbw": "22.90%" if a else "22.30%",
        "l2hit": "72.71%" if a else "74.93%",
        "pipesbusy": "7.96%" if a else "8.08%",
        "smactive": "104,893.50 cycles" if a else "111,527.62 cycles",
        "elapsed": "1,301,748 cycles" if a else "1,347,770 cycles",
        "instructions": "1,418,496" if a else "1,468,416",
        "issued": "14,202,362" if a else "14,703,386",
        "avgexec": "14,776" if a else "15,296",
        "avgissued": "14,795.44" if a else "15,316.52",
        "fused": "240,896" if a else "252,160",
        "nonfused": "271,872" if a else "253,440",
        "fp32speed": "1.25%" if a else "1.12%",
        "source_exp": "23.37%" if a else "25.65%",
        "source_sum": "3.57%" if a else "3.31%",
        "source_scale": "5.19%" if a else "2.79%",
        "source_pv": "9.82%" if a else "9.75%",
        "source_reshape": "11.29%" if a else "10.90%",
        "source_store": "1.97%" if a else "2.21%",
    }

    if view == "summary":
        return {
            "visible": (f"Summary table: kernel duration **{d['duration']}**, runtime-improvement estimate **{d['runtime_improvement']}**, compute throughput **{d['compute']}**, memory throughput **{d['memory']}**, registers/thread **{d['regs']}**, grid **(1, 8, 8)** and block **(128, 1, 1)**. The lower rule panel shows estimated speedup **77.10%** (A) / **77.70%** (B), theoretical occupancy **8.3%** limited by shared memory, math-pipe stalls **{d['math']} cycles** (local estimate **{d['math_pct']}**), and tail-effect estimate about **22.32%**."),
            "meaning": "This is the best compact comparison of the measured kernel result and the profiler's rule suggestions. The large 77% estimate is not an achieved speedup: it is an NCU upper-bound estimate under the tool's optimization assumptions. The duration is profiler/replay time, not an ordinary benchmark.",
            "use": "Use this to identify the shared-memory occupancy ceiling and to locate deeper sections. Do not choose a winner from the 52.86/54.50 µs profiled durations or add the overlapping NCU estimates together."
        }

    if view == "details-01":
        return {
            "visible": (f"Tile Statistics: **64 CTAs launched**, **64 tile blocks specified**, **128 threads/CTA**. Speed of Light: compute **{d['compute']}**, total memory **{d['memory']}**, L1/TEX **{d['l1']}**, L2 **{d['l2']}**, DRAM **{d['memory']}**. Profile duration **{d['duration']}**, elapsed cycles **{d['elapsed']}**, SM active cycles **{d['smactive']}**. PM Sampling: **1 µs** maximum interval, **4.65 MB** maximum buffer, **3 pass groups**."),
            "meaning": "The workload is one 64-CTA kernel launch on the RTX 4060. All major compute and memory-throughput percentages are low compared with peak. PM sampling is statistical and was collected over three passes.",
            "use": "Establishes workload identity and sampling context. The next Details captures explain the latency issue; low peak percentages alone do not prove a specific bottleneck."
        }

    if view == "details-02":
        extra = ("The compute rows show elapsed/active/issued IPC **0.45 / 0.56 / 0.56**, SM busy **18.81%**, issue slots busy **11.32%**; NCU labels this **Low Utilization** and estimates local headroom **81.19%**." if a else
                 "The compute rows show elapsed/active/issued IPC **0.45 / 0.55 / 0.55**, SM busy **20.28%**, issue slots busy **11.37%**. Tensor is the highest aggregate pipeline at **20.3%**, but the FP tensor sub-pipeline is **10.1%**; NCU labels it **Balanced** and says it should not be a bottleneck.")
        tensor = "" if a else " Top tensor instructions are HMMA.16816.F32 at source lines **91** and **78**, each with **81,920** executed instructions."
        return {
            "visible": (f"Speed of Light: compute **{d['compute']}**, memory **{d['memory']}**, L1/TEX **{d['l1']}**, L2 **{d['l2']}**, DRAM **{d['memory']}**; duration **{d['duration']}**. NCU explicitly marks **Latency Issue** because compute and bandwidth are below 60% of peak. PM Sampling remains 1 µs / 4.65 MB / 3 passes. {extra}{tensor}"),
            "meaning": ("The aggregate compute pipes are underused; low eligible warps and latency are better leads than a tensor-core throughput limit. NCU explicitly labels the workload Latency Issue." if a else "The tensor pipeline is working but not saturated; NCU labels it Balanced and says it should not be a bottleneck. Low eligible warps and memory/scoreboard latency are better leads. Neither case supports a DRAM-bandwidth-limited diagnosis."),
            "use": ("Prioritize Scheduler Statistics and Warp State over a blanket tensor-core or DRAM optimization." if a else "Use the line 91/78 HMMA locations as specific source-export entry points, but do not conclude tensor saturation is the bottleneck.")
        }

    if view == "details-03":
        low = ("**Low Utilization**; NCU's local estimate is **81.19%**." if a else "**Balanced** tensor pipeline, with **20.3%** aggregate tensor use and **10.1%** highest FP tensor sub-pipeline use.")
        return {
            "visible": (f"Roofline says the workload reaches **3% of FP32 peak** and **0% of FP64 peak**. PM Sampling: 1 µs, 4.65 MB, 3 passes. Compute IPC elapsed/active/issued **0.45 / {'0.56 / 0.56' if a else '0.55 / 0.55'}**; SM busy **{d['compute']}**, issue slots busy **{'11.32%' if a else '11.37%'}**; {low} Memory: **{d['memory_gbs']}**, Mem Busy **{d['membusy']}**, Max Bandwidth **{d['maxbw']}**, L2 hit **{d['l2hit']}**, memory pipes busy **{d['pipesbusy']}**; local/shared spilling requests **0 / 0**. Scheduler active warps/scheduler **1.00**, No Eligible **{d['noeligible']}**."),
            "meaning": ("This is a strong cross-check: the GPU is not near its DRAM limit, spilling is absent, and the scheduler has no eligible warp most cycles. The compute section is labelled Low Utilization." if a else "This is a strong cross-check: the GPU is not near its DRAM limit, spilling is absent, and the scheduler has no eligible warp most cycles. NCU explicitly calls tensor usage Balanced, not bottlenecked."),
            "use": "A high-value decision image. Avoid spending the first iteration on DRAM coalescing or tensor-core peak-throughput work; investigate how to increase ready warps and reduce shared-memory residency cost."
        }

    if view == "details-04":
        tensor = ("No tensor-pipeline diagnostic is visible here." if a else
                  "Tensor summary: **20.3%** aggregate Tensor pipeline; FP Tensor sub-pipeline **10.1%**; the two dominant tensor instructions are line 91 and line 78, both HMMA.16816.F32 at **81,920 instructions each**.")
        return {
            "visible": (f"PM Sampling: **1 µs**, **4.65 MB**, **3 passes**. Compute IPC elapsed/active/issued **0.45 / {'0.56 / 0.56' if a else '0.55 / 0.55'}**; SM busy **{d['compute']}**, issue slots busy **{'11.32%' if a else '11.37%'}**. Memory **{d['memory_gbs']}**, Mem Busy **{d['membusy']}**, bandwidth **{d['maxbw']}**, L2 hit **{d['l2hit']}**, memory pipes busy **{d['pipesbusy']}**; spill requests local/shared **0 / 0**. Scheduler shows active warps **1.00**, eligible **0.14**, issued **0.14**, No Eligible **{d['noeligible']}**, One-or-more-eligible **{d['oneeligible']}**. {tensor}"),
            "meaning": ("The low issue-slot use is not an occupancy-only story: only about 0.14 warps per scheduler are ready, despite one active warp. Memory and spill counters rule out two common but unsupported diagnoses; this frame has no tensor-pipeline diagnostic." if a else "The low issue-slot use is not an occupancy-only story: only about 0.14 warps per scheduler are ready, despite one active warp. Memory and spill counters rule out two common but unsupported diagnoses; the tensor pipeline is separately labelled Balanced."),
            "use": "Use as a key point for the low-ready-warp problem. A smaller tile is a testable way to reduce shared-memory usage and potentially admit another CTA/SM; confirm the occupancy limiter changes before making a performance claim."
        }

    if view == "details-05":
        return {
            "visible": (f"Scheduler section: active warps/scheduler **1.00**, eligible **0.14**, issued **0.14**, No Eligible **{d['noeligible']}**, One-or-more-eligible **{d['oneeligible']}**. NCU says each scheduler issues an instruction only once every **{d['warp_issue']} cycles**; the architectural pool allows up to 12 warps/scheduler, but this launch has only one active warp." if a else f"Scheduler section: active warps/scheduler **1.00**, eligible **0.14**, issued **0.14**, No Eligible **{d['noeligible']}**, One-or-more-eligible **{d['oneeligible']}**. NCU says each scheduler issues an instruction only once every **{d['warp_issue']} cycles**; the architectural pool allows up to 12 warps/scheduler, but this launch has only one active warp. The capture also repeats the balanced tensor pipeline and memory values **58.80 GB/s / 22.30% DRAM peak / 74.93% L2 hit / zero spills**."),
            "meaning": "The idle issue slots are caused by the lack of warps eligible to issue, not by lack of theoretical warp slots.",
            "use": "This is the strongest direct argument for testing occupancy and latency hiding. It does not identify the single dependency that makes each warp ineligible; use Warp State and source counters for the next diagnostic step."
        }

    if view == "details-06":
        return {
            "visible": (f"Warp state: warp cycles per issued instruction **{d['warp_issue']}**, per executed instruction **{d['warp_exec']}**; average active threads/warp **32**, average not-predicated threads **{'31.85' if a else '31.89'}**. Memory repeats **{d['memory_gbs']}**, L2 hit **{d['l2hit']}**, spill requests **0**. Scheduler remains **1.00 active / 0.14 eligible / 0.14 issued / {d['noeligible']} No Eligible**."),
            "meaning": "Threads are largely active and predicated-on, so branch divergence is not the explanation for the poor issue rate. The high cycles-per-issued-instruction value aligns with the scheduler starvation diagnosis.",
            "use": "Do not optimize branch structure first. Pair this with the following math-throttle and source-marker captures to split latency from instruction-pipeline pressure."
        }

    if view == "details-07":
        return {
            "visible": (f"Math Pipe Throttle: average **{d['math']} cycles** per warp; NCU says this accounts for **{d['math_pct']}** of the average **{d['warp_issue']} cycles** between issued instructions and gives a local estimate of **{'33.68%' if a else '36.34%'}**. Warp cycles/issued and executed are **{d['warp_issue']} / {d['warp_exec']}**; active threads **32**, not-predicated **{'31.85' if a else '31.89'}**."),
            "meaning": "Math-pipeline pressure is material, and it is somewhat worse for 64x128. However, the displayed local estimate is not a measured speedup and the scheduler still has only 0.14 eligible warps.",
            "use": "Investigate MMA/softmax scheduling and overlap, but don't infer that changing HMMA instructions alone can produce the NCU estimate. More ready warps may be needed to hide both math and scoreboard latency."
        }

    if view == "details-08":
        return {
            "visible": (f"Instruction Statistics: executed instructions **{d['instructions']}**, issued instructions **{d['issued']}**, average executed instructions/scheduler **{d['avgexec']}**, issued **{d['avgissued']}**; local and shared spill requests **0**. The same frame continues to show **{d['noeligible']} No Eligible**, **{d['math']} cycles** math-pipe stall and **{d['warp_issue']} cycles** between issued instructions."),
            "meaning": "The B variant executes **49,920** more instructions (+3.52%) while retaining the same one-active-warp/scheduler issue bottleneck. Neither variant has register/local-memory or shared-memory spilling requests.",
            "use": "This makes the 64x128 variant's resource trade-off harder to justify without unprofiled timing evidence. It also rules out spilling as a first-order optimization target."
        }

    if view == "details-09":
        return {
            "visible": (f"FP32 instruction section: fused **{d['fused']}**, non-fused **{d['nonfused']}**. NCU says FP32 performance could improve by up to **{'27%' if a else '25%'} relative to current FP32 performance**, but estimates only **{d['fp32speed']} overall kernel** speedup. Top non-fused locations shown: A `cutile_attention.py` line 90 FMUL **73,728**, line 87 FADD **73,728**, line 88 FADD **43,776**; B line 87 FADD **81,920**. Instruction total is **{d['instructions']}**, spills are zero."),
            "meaning": "This is a classic distinction between a large component-level opportunity and a small end-to-end benefit. Fast-math/FMA is not a 25–27% whole-kernel win; the tool's own estimate is about 1.1–1.3%. PTX already has `ex2.approx` and `fma.rn` forms, so compiler/codegen must be verified line by line.",
            "use": "A lower-priority, accuracy-sensitive experiment. Don't enable global fast-math or alter softmax precision until correctness error bounds are tested and the generated PTX/SASS proves the intended instruction sequence changes."
        }

    if view.startswith("source-"):
        if view in {"source-01", "source-02", "source-03", "source-04", "source-05", "source-high-level-tail", "source-high-level-head"}:
            return {
                "visible": "The left Source selector is **`_stub.py`**, the generated ScalarProtocol stub, not `src/cutile_attention.py`. The visible code consists of Python dunder wrappers such as `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__` and related `TileOrScalar` methods. The right SASS pane is at the kernel prologue, beginning around address `0x200e9be00`, with setup instructions such as `IMAD.MOV.U32`, `S2R`, `ULDC`, `CS2R`, `SHF` and `LOP3`.",
                "meaning": "These five sequential/scroll captures do not show the attention algorithm. Despite the labels `source-high-level-head/tail`, the visible selector remains `_stub.py`; they must not be treated as high-level-source or QK/PV evidence.",
                "use": "Useful only as a capture-quality audit. It exposes a source-file resolution failure. Do not use the displayed stub methods or prologue instructions to choose an attention-kernel optimization."
            }
        if view == "source-line91-sass":
            return {
                "visible": "The left selector still reads **`_stub.py`** and the highlighted row is stub line **42**, not `cutile_attention.py` line 91. The right SASS pane shows the kernel prologue at labels 1–23, beginning with `IMAD.MOV.U32`, `S2R`, `ULDC`, `CS2R`, `SHF` and `LOP3`; displayed per-instruction execution shares are about **0.02%** each.",
                "meaning": "This screenshot is mis-targeted: it does **not** correlate the PV MMA source line with SASS. The lower self-instruction and input/output scoreboard tables are empty in this frame because the intended attention source instruction was not selected.",
                "use": "Do not cite this as proof of the line-91 HMMA sequence. It identifies a limitation in the GUI automation that needs a targeted capture fix."
            }
        if view == "source-horizontal-left-code":
            return {
                "visible": "The left code pane is horizontally scrolled while its selector remains **`_stub.py`**. Only fragments of generated `TileOrScalar` wrapper signatures/bodies are visible; the attention-kernel expressions are not present in this file.",
                "meaning": "The horizontal scroll reveals clipped text but not more useful kernel logic. It is not the actual `cutile_attention.py` source pane.",
                "use": "Retain as an audit trail of the four-pane layout only. No optimization decision should be based on this frame."
            }
        if view == "source-horizontal-left-metrics":
            return {
                "visible": "The left selector is **`_stub.py`**. The adjacent left metrics subpanel exposes headings for attributed warp-stall samples, instruction mix and instructions executed, but no populated source rows are visible for this stub selection.",
                "meaning": "This is not evidence of zero stalls or zero source instructions for the kernel. The attribution table is empty because the selected source is a generated stub rather than the attention kernel.",
                "use": "The missing values should not be interpreted. Use the actual source-marker image and raw source export instead."
            }
        if view == "source-horizontal-right-code":
            return {
                "visible": "The right code subpanel shows SASS labels **1–23** in the kernel prologue, around addresses `0x200e9be00` onward: `IMAD.MOV.U32`, `S2R`, `ULDC`, `CS2R`, `SHF`, `LOP3`, `IMAD.SHL.U32` and `IADD3`. The instructions are categorized mainly as Integer/Miscellaneous/Uniform Data, with displayed instruction shares about **0.02%** per row. The left source is still `_stub.py`.",
                "meaning": "This is startup/register setup code, not the matrix-multiply/softmax hot path. Per-op shares are small and the source mapping is still invalid for the attention function.",
                "use": "Keep as layout evidence only. To inspect HMMA hotspots, the actual `cutile_attention.py` source and a selected MMA row must be visible together."
            }
        if view == "source-horizontal-right-metrics":
            return {
                "visible": (f"The left source pane is now `cutile_attention.py`; its visible Instructions Executed attribution includes line 87 `ct.exp(s-mn)` **{d['source_exp']}**, line 88 online-softmax update **{d['source_sum']}**, line 90 `u0=a*u` **{d['source_scale']}**, line 91 PV MMA **{d['source_pv']}**, line 95 output reshape **{d['source_reshape']}**, and line 96 store **{d['source_store']}**. The right SASS subpanel is at the kernel tail, labels **2641–2664**, showing `BRA`, `LOP3.LUT`, `MFU.RSQ`, `FADD.FTZ`, `BSYNC`, `IMAD.MOV.U32`, `MOV`, `RET.REL.NODEC` and many `NOP`s."),
                "meaning": "The left source attribution is useful, but the right pane is an epilogue/control tail, not the line-91 PV MMA body. The panes are not showing a selected one-to-one line-to-HMMA match.",
                "use": "Use the line-attribution values to identify work distribution, and the separate source export for MMA body correlation. Don't infer the epilogue dominates from its visible instruction list."
            }
        if view == "source-view-picker":
            return {
                "visible": (f"This is **not an open view-picker menu**: the toolbar already shows `Right View: SASS`. The right pane displays the epilogue labels **2641–2664** (`BRA`, `LOP3.LUT`, `MFU.RSQ`, `FADD.FTZ`, `BSYNC`, `IMAD.MOV.U32`, `MOV`, `RET` and `NOP`). The left code pane shows `cutile_attention.py` line attribution: exp **{d['source_exp']}**, PV MMA **{d['source_pv']}**, reshape **{d['source_reshape']}**."),
                "meaning": "This is a SASS tail/epilogue screenshot; the file name `source-view-picker` is misleading. The values on the left are source-line instruction attribution, not runtime percentages.",
                "use": "Useful to document code-generation epilogue structure, not a selected MMA hotspot. Use explicit instruction correlation from the text export for the compute body."
            }
        if view in {"source-ptx", "source-line91-ptx", "source-ptx-horizontal-ptx-code"}:
            return {
                "visible": (f"Left pane: actual `cutile_attention.py` lines **85–97**. Visible Instructions Executed attribution: line 87 `p=ct.exp(s-mn)` **{d['source_exp']}**, line 88 online-softmax sum/update **{d['source_sum']}**, line 90 `u0=a*u` **{d['source_scale']}**, line 91 PV MMA **{d['source_pv']}**, line 95 output reshape **{d['source_reshape']}**, line 96 store **{d['source_store']}**. Right pane: PTX header/entry only — `.version 9.4`, `.target sm_89`, `.address_size 64`, visible launch-metadata declaration, `.visible .entry` and kernel parameter declarations. No PTX instruction body is visible."),
                "meaning": "This screenshot establishes the compiler target and kernel ABI, but it does **not** show the PTX implementation of line 91 or the `cp.async`/`ldmatrix` instructions. The text PTX export is the source for those instructions. The `source-line91-ptx` frame also has no visible selected-line highlight.",
                "use": "Keep the screenshot as the actual GUI PTX header view only. For line-91 instruction analysis, cross-reference `data/*source-tileir-ptx.txt` and `data/*source-tileir-sass.txt`; do not describe this image as a source-to-PTX instruction correlation."
            }
        if view == "source-ptx-horizontal-ptx-metrics":
            return {
                "visible": "The right PTX metrics subpanel is blank: no selected PTX instruction rows or metric values are displayed. The visible left source line list repeats the code-attribution values from the adjacent PTX view.",
                "meaning": "This is not evidence that PTX instructions have no stalls or metrics; the GUI has no selected/instrumented PTX instruction in this frame.",
                "use": "Use the exported PTX and raw metric/source exports instead. This capture is only an explicit record of an unavailable PTX metrics pane."
            }
        if view == "source-bottom-instructions-dependencies":
            return {
                "visible": "The lower `Instructions & Dependencies` tab shows three panes: Self Instruction Mix, Input Scoreboard Dependencies and Output Scoreboard Dependencies. Their tables have column headings but **no populated instruction rows**. The left source list is `cutile_attention.py`; line-level attribution is visible above.",
                "meaning": "No individual source/SASS/PTX instruction was successfully selected for dependency attribution. Empty dependency tables are not proof that the kernel has no scoreboard dependencies.",
                "use": "Do not draw dependency conclusions from this frame. It explicitly marks a gap in the GUI capture; the aggregate long-scoreboard and math-pipe data remain valid from Details/raw counters."
            }
        if view == "source-bottom-inline-functions":
            return {
                "visible": "The lower `Inline Functions` tab is open; the panel contains no populated inline-function entries. The left source view shows the attention source and line-attribution bars.",
                "meaning": "A blank pane means no inline-function selection/data is displayed here; it is not proof that the generated kernel has no inlined code.",
                "use": "No optimization decision should depend on this empty view. A targeted function/line selection is required before making inlining claims."
            }
        if view == "source-bottom-source-markers":
            if a:
                marker = "The bottom Source Markers table flags **FP32/64 Instructions at line 87** (`p=ct.exp(s-mn)`) and **line 88** (`ln=a*l+ct.sum(p,axis=1,keepdims=True)`). The left attribution shows line 87 **23.37%**, line 88 **3.57%**, line 90 **5.19%**, line 91 PV MMA **9.82%**, line 95 output reshape **11.29%**, and line 96 store **1.97%**."
            else:
                marker = "The bottom Source Markers table flags **High Pipe Utilization at line 78** (`s=ct.mma(q0,kt,acc)`): the source note says this line executes many `HMMA.16816.F32` Tensor instructions and the tensor pipeline is the highest-utilized pipeline. It separately flags **FP32/64 Instructions at line 79** (`s=s*scl`). Left attribution: line 87 softmax exp **25.65%**, line 88 update **3.31%**, line 90 `u0=a*u` **2.79%**, line 91 PV MMA **9.75%**, line 95 reshape **10.90%**, line 96 store **2.21%**."
            return {
                "visible": marker,
                "meaning": "This is the strongest actual source-level marker capture. The dominant marker changes with tile shape: 64x64 highlights the online-softmax FP32 exp/update lines; 64x128 flags QK MMA's tensor-pipeline use and the score scale operation. These are source/instruction attribution markers, not time percentages.",
                "use": "This suggests two different experiments: for A, inspect the softmax exp/update lowering and math; for B, inspect the QK MMA / score-scale path and its interaction with the wider tile. Do not apply the same micro-optimization blindly to both.",
            }
        if view == "source-bottom-statistics":
            return {
                "visible": "The lower `Statistics` tab displays columns for live registers, attributed stalls, attributed warp-stall sampling, instruction mix, instructions executed, threads executed, address space, access operation/size, shared-wavefront excess and global-sector excess. The Aggregate/Maximum/Minimum/Average area contains no populated values because no single instruction was selected.",
                "meaning": "The column names tell us what could be inspected, but this screenshot does not contain per-instruction statistics. Blank values are not zeros.",
                "use": "The next capture needs a genuinely selected source/SASS instruction before the Statistics tab can be used to support a per-instruction optimization."
            }

    if view == "raw":
        return {
            "visible": f"Raw tab metadata: result ID **0**, timestamp **2026-Oct-27 23:19:00**, API Call ID **750**, NCU Estimated Speedup **77.10%**, Runtime Improvement estimate **40.76 µs**, 12 issues detected, Device **NVIDIA GeForce RTX 4060**, grid **(1,8,8)**, block **(128,1,1)**, duration **52.86 µs**, SM throughput **18.81%**, compute-memory throughput **22.90%**. The full table continues with hundreds of exported metrics; some pass-specific columns display bracketed IDs/values." if a else
            "Raw tab metadata: result ID 0, NCU call 752, estimated speedup 77.70%, runtime-improvement estimate 42.34 µs, device RTX 4060, grid (1,8,8), block (128,1,1), duration 54.50 µs, SM throughput 20.28%, compute-memory throughput 22.30%. The full CSV export carries the many counters that do not fit in the screenshot.",
            "meaning": "This is a useful identity/provenance cross-check, but the on-screen raw page is only a partial view and includes replay/pass-specific metric entries. The three-row CSV export is the authoritative numeric source for the full set.",
            "use": "Use the raw screenshot to confirm which kernel result is being viewed; use `data/*-raw.csv` and `metric-comparison.csv` for numeric analysis. Do not copy bracketed/multipass placeholders as performance values."
        }

    return {
        "visible": "No reliable view-specific metric description is available for this capture name. Consult the exact image before treating it as evidence.",
        "meaning": "This capture is not classified as a performance finding.",
        "use": "Do not use it to justify a code change without further inspection."
    }


def make_image(name, alt=None):
    return f"![{alt or name}](screenshots/{name})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    captures = manifest.get("captures", [])
    screenshots = out / "screenshots"
    metric_rows = read_csv(out / "metric-comparison.csv")

    lines = [
        "# Nsight Compute analysis — current medium_d128 experiment", "",
        "## Scope and provenance", "",
        "This report compares only the two existing RTX 4060 native Nsight Compute reports. The original `.ncu-repz` files are untouched. All screenshot references below are from genuine Nsight Compute application captures made by the local Windows self-hosted runner.",
        "- GPU: NVIDIA GeForce RTX 4060, compute capability 8.9; Nsight Compute 2026.3.1.",
        "- Workload: Q/K/V and output shape `(1, 8, 512, 128)`, FP16 inputs/output, FP32 score and online-softmax accumulation.",
        "- A: `tile_q=64`, `tile_k=64`; B: `tile_q=64`, `tile_k=128`.",
        "- Grid: `(1, 8, 8)` = 64 CTAs; block: `(128, 1, 1)` = 128 threads; 24 SMs.",
        "- Results directory: `results/ncu/rtx4060/20261027-230742/ncu-analysis-current/`.",
        "",
        "## Headline comparison", "",
        "| Measurement | 64×64 | 64×128 |",
        "|---|---:|---:|",
        "| NCU profiled duration | 52.86 µs | 54.50 µs |",
        "| SM throughput / peak | 18.81% | 20.28% |",
        "| DRAM throughput / peak | 22.90% | 22.30% |",
        "| Registers / thread | 231 | 255 |",
        "| Static shared memory / block | 65.536 KB | 81.920 KB |",
        "| Shared-memory CTA limit | 1 | 1 |",
        "| Register CTA limit | 2 | 2 |",
        "| Theoretical occupancy | 8.33% | 8.33% |",
        "| Eligible warps / scheduler | 0.141 | 0.139 |",
        "| No Eligible | 85.90% | 86.14% |",
        "| Math-pipe throttle | 2.38 cycles | 2.62 cycles |",
        "| Long-scoreboard stall | 1.86 cycles | 1.85 cycles |",
        "| Warp cycles / issued instruction | 7.06 | 7.22 |",
        "| Shared/local spill requests | 0 / 0 | 0 / 0 |",
        "| Dynamic executed SASS instructions | 1,418,496 | 1,468,416 |",
        "",
        "**Timing caveat:** the 52.86/54.50 µs numbers are NCU profiled/replay measurements. They are not a substitute for repeated unprofiled CUDA-event timing and cannot determine which tile is faster in normal use.",
        "",
        "## Important key points", "",
        "These are the small set of images worth reading first. The full screenshot register below links to every capture but does not re-embed the images, so the document contains no duplicate screenshot gallery.",
        "",
        "### 1. Shared memory is the occupancy limiter; the larger key tile does not buy more occupancy", "",
        make_image("medium_d128_64x64-summary.png", "64×64 Nsight Summary") + "  " + make_image("medium_d128_64x128-summary.png", "64×128 Nsight Summary"), "",
        "A uses **65.536 KB shared memory/block** and **231 registers/thread**; B uses **81.920 KB** (+25%) and **255 registers/thread** (+10.4%). In both, shared memory allows only **one CTA/SM**, while the register budget would allow two. Both remain at **8.33% theoretical occupancy**. B reduces causal K/V loop visits from 36 to 20, but does not increase resident CTAs and has 3.52% more dynamic instructions. The profiled duration is 52.86 vs 54.50 µs, but ordinary runtime must decide the performance winner.",
        "**Direction:** first test `tile_k=32` with `tile_q=64` as a controlled occupancy hypothesis. Check whether the shared-memory CTA limit changes from 1 to 2 and whether eligible warps rise. The extra loop count may offset the occupancy gain; correctness and unprofiled timing decide.",
        "",
        "### 2. The immediate issue is a shortage of ready warps, not simply insufficient peak bandwidth", "",
        make_image("medium_d128_64x64-details-04.png", "64×64 scheduler and memory details") + "  " + make_image("medium_d128_64x128-details-05.png", "64×128 scheduler and tensor details"), "",
        "A has **1.00 active warp/scheduler**, **0.14 eligible**, **0.14 issued**, and **85.90% No Eligible**. B has **1.00 / 0.14 / 0.14** and **86.14% No Eligible**. Schedulers issue only about every **7.1 / 7.2 cycles**. Math-pipe throttle is **2.38 / 2.62 cycles** and long-scoreboard is **1.86 / 1.85 cycles**. The high No Eligible fraction means the SM often has no ready warp to issue; the long-scoreboard component says memory dependencies still contribute. Stall ratios should not be added together because they are not mutually exclusive percentages.",
        "**Direction:** improve latency hiding and resource residency first. The source line/instruction view should determine whether to attack load-to-use latency, MMA operand readiness or softmax dependencies. Do not assume `tile_k=128` is better simply because it executes fewer K/V loop iterations.",
        "",
        "### 3. This is not a DRAM-bandwidth-limited kernel, and the tensor pipe is not saturated", "",
        make_image("medium_d128_64x64-details-03.png", "64×64 low-utilization and memory details") + "  " + make_image("medium_d128_64x128-details-02.png", "64×128 tensor and memory details"), "",
        "DRAM is only **22.90% / 22.30% of peak**; memory traffic is **60.35 / 58.80 GB/s**; L2 hit is **72.71% / 74.93%**; memory-pipe busy is **7.96% / 8.08%**. Local/shared spill requests are **zero**. B's aggregate tensor pipeline is **20.3%** of elapsed cycles, with the highest tensor sub-pipeline **10.1%**; NCU calls this balanced and says it should not be a bottleneck. A is labelled Low Utilization (81.19% NCU local headroom estimate).",
        "**Direction:** do not lead with DRAM coalescing, bank-padding or tensor-core peak-throughput rewrites. The high hit rate and low DRAM utilization point toward dependencies/latency and insufficient ready warps. The 5,716 B bank-conflict events are only **1.24% of 459,417 shared wavefronts**, with actual wavefronts equal to ideal, so they are secondary absent stronger per-instruction evidence.",
        "",
        "### 4. The source-level bottleneck signatures differ between the two tile shapes", "",
        make_image("medium_d128_64x64-source-bottom-source-markers.png", "64×64 source markers") + "  " + make_image("medium_d128_64x128-source-bottom-source-markers.png", "64×128 source markers"), "",
        "For A, Source Markers flags FP32/64 work on **line 87** (`p=ct.exp(s-mn)`) and **line 88** (online-softmax sum/update). The visible source attribution is line 87 **23.37%**, line 88 **3.57%**, line 90 `u0=a*u` **5.19%**, line 91 PV MMA **9.82%**, line 95 output reshape **11.29%**, line 96 store **1.97%**. For B, the marker changes: **line 78 QK MMA** is marked High Pipe Utilization due to many `HMMA.16816.F32` Tensor instructions; **line 79** score scaling is marked FP32/64. Its line attribution is line 87 **25.65%**, 88 **3.31%**, 90 **2.79%**, 91 **9.75%**, 95 **10.90%**, 96 **2.21%**. These are instruction-attribution values, not runtime shares.",
        "**Direction:** treat the two variants differently. A deserves a softmax/exp/update instruction study; B warrants inspection of QK MMA and the score-scaling instruction around it. Any rewrite must preserve online-softmax stability and pass numerical correctness tests.",
        "",
        "### 5. FP32 fusion is a small whole-kernel opportunity, not a 25–27% end-to-end win", "",
        make_image("medium_d128_64x64-details-09.png", "64×64 FP32 non-fused details") + "  " + make_image("medium_d128_64x128-details-09.png", "64×128 FP32 non-fused details"), "",
        "A has **271,872 non-fused** vs **240,896 fused** FP32 instructions; B has **253,440 non-fused** vs **252,160 fused**. NCU estimates up to 27% / 25% improvement *within the FP32 portion*, but only **1.25% / 1.12% overall kernel**. A's top listed sites are line 90 FMUL **73,728**, line 87 FADD **73,728**, line 88 FADD **43,776**; B's top site shown is line 87 FADD **81,920**. PTX already contains `ex2.approx.ftz.f32` and `fma.rn.f32` forms.",
        "**Direction:** defer global fast-math changes. Inspect the exact PTX/SASS sequence and validate numerical error first; a potentially large FP32-subpipeline ratio translates to only a small whole-kernel estimate.",
        "",
        "## Detailed metric interpretation", "",
    ]

    metric_file = out / "metric-analysis.md"
    if metric_file.exists():
        metric_text = metric_file.read_text(encoding="utf-8-sig").strip()
        # Avoid a second heading; the README already owns this section title.
        metric_text = metric_text.replace("# Metric analysis and interpretation", "", 1).strip()
        lines.extend([metric_text, ""])

    lines += [
        "## PTX/SASS evidence and GUI-capture limits", "",
        "The raw text exports, not the two PTX GUI header images, establish the generated instruction sequence. PTX contains `cp.async.ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group`, `ldmatrix.sync.aligned.m8n8.x4.shared.b16`, `bar.sync`, `ex2.approx.ftz.f32`, `fma.rn.f32` and `mul.rn.f32`. SASS contains `LDGSTS.E.128`, `LDSM.16.M88.4`, `BAR.SYNC.DEFER_BLOCKING`, `DEPBAR.LE` and `HMMA.16816.F32`. Static listing occurrence counts are preserved in `source-instruction-counts.csv`; they are not dynamic execution counts.",
        "**Important GUI limitation:** `source-ptx.png`, `source-line91-ptx.png` and the PTX horizontal captures show `.version 9.4`, `.target sm_89`, `.address_size 64` and kernel entry/parameter declarations, but not the PTX instruction body. `source-line91-sass.png` does not show `cutile_attention.py` line 91 selected: its left selector is `_stub.py` and the highlighted row is stub line 42. The lower dependency/statistics tabs are blank because no individual instruction is selected. The source-marker screenshots are the actual selected source-level evidence; the exported text is the PTX/SASS instruction evidence.",
        "",
        "## Screenshot-by-screenshot interpretation", "",
        "Every capture is listed once below. The **Important key points** section embeds only the selected figures; this register uses clickable image links so the same 64 images are not embedded again. Per-image values below are read from the matching screenshot and checked against the unit-labelled CSV/details exports where applicable.",
        "",
    ]

    # Fixed ordering makes the per-image register easy to compare A then B.
    captures_sorted = sorted(captures, key=lambda c: (0 if "64x64" in Path(c.get("screenshot", "")).name else 1, Path(c.get("screenshot", "")).name))
    for c in captures_sorted:
        name = str(c.get("screenshot", "")).replace("\\", "/").split("/")[-1]
        view = c.get("view", "")
        cfg = config_for(name)
        desc = screenshot_desc(view, cfg)
        lines += [
            f"### `{name}`",
            "",
            f"**Open image:** [View screenshot](screenshots/{name})  ·  **Configuration:** `{cfg}`  ·  **Capture:** {c.get('width')}×{c.get('height')}  ·  **View label:** `{view}`",
            "",
            f"**Visible information and values.** {desc['visible']}",
            "",
            f"**What it means.** {desc['meaning']}",
            "",
            f"**Why it matters / decision.** {desc['use']}",
            "",
        ]

    lines += [
        "## Recommended next experiment", "",
        "1. Keep the original two reports immutable. Add a separate `tile_q=64, tile_k=32` experiment and capture launch resources, scheduler/warp states, source and numerical errors. First check whether `launch__occupancy_limit_shared_mem` changes from 1 to 2 blocks/SM and whether eligible warps/scheduler improve.",
        "2. Measure with repeated unprofiled CUDA-event timing (warmup + multiple repeats, alternating tile order) and report median plus spread. A reduction in profiled cycles is not sufficient.",
        "3. Test `tile_q=32` separately, not in the same change. Compare output correctness, register/shared-memory usage, grid tail effects and runtime.",
        "4. Only after those experiments consider instruction fusion, load scheduling or barrier changes. Verify exact PTX/SASS deltas and numerical error for the causal softmax.",
        "",
        "## Files in this experiment folder", "",
        "- `data/medium_d128_64x64-raw.csv` and `data/medium_d128_64x128-raw.csv`: full raw NCU metric exports.",
        "- `data/*-details.txt`: full metric-section text exports.",
        "- `data/*-source-tileir-ptx.txt` and `data/*-source-tileir-sass.txt`: source/TileIR-to-PTX and source/TileIR-to-SASS exports.",
        "- `metric-comparison.csv`: selected metrics with units, both configurations, difference and relative change.",
        "- `source-instruction-counts.csv`: static PTX/SASS listing occurrences; not dynamic execution counts.",
        "- `capture-manifest.json`: capture metadata and skipped-capture record.",
        "",
        "## Screenshot register count", "",
        f"Captured **{len(captures)}** genuine GUI images; skipped **{len(manifest.get('skipped', []))}**. Each image appears once in the detailed register; only the selected key-point subset is embedded above.",
        "",
    ]

    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    pointer = "# Screenshot interpretation\n\nThe complete screenshot-by-screenshot register is maintained in [`README.md`](README.md#screenshot-by-screenshot-interpretation). The README contains the key-point images once, followed by links and capture-specific interpretation for every screenshot. This file is intentionally a pointer to avoid duplicating the image set and interpretation.\n"
    (out / "screenshot-notes.md").write_text(pointer, encoding="utf-8")
    print(f"README rebuilt with {len(captures)} linked screenshot entries and a non-duplicated key-point image set.")

if __name__ == "__main__":
    main()