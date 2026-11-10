# medium_d64-tq64-tk64-lat0

**Workload:** medium_d64 | **TQÃ—TK:** 64Ã—64 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report medium_d64-tq64-tk64-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 24.51 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 10.56 Âµs (0.011 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 8, 4) / (128, 1, 1) | 32 CTAs; 4,096 threads. |
| Compute / memory / DRAM throughput | 13.89% / 30.42% / 30.42% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 15.84% / 11.86% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 79.76 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 55.28% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 11.86% / 7.74% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 49.15 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 16.67% / 10.35% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4.97 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 2 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 80.84% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.21 / 0.19 / 1.23 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 6.40 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 80.84% and Eligible Warps Per Scheduler is 0.21. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 49.15 KiB static shared memory per CTA; the shared-memory block limit is 2 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 30.42% and bandwidth 79.76 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 69.58% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 5.2 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.23 active warps per scheduler, but only an average of 0.21 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons on the Warp State Statistics and Source Counters sections. |
| TheoreticalOccupancy | 69.58% | The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory. |
| WorkloadImbalance | 15.36% | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 21.19% above the average, while the minimum instance value is 24.37% below the average. |
| WorkloadImbalance | 14.77% | One or more SMs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 20.48% above the average, while the minimum instance value is 22.40% below the average. |
| WorkloadImbalance | 14.77% | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Additionally, other L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 20.48% above the average, while the minimum instance value is 22.40% below the average. |

## 5. PTX/SASS evidence
- PTX export: medium_d64-tq64-tk64-lat0/source-tileir-ptx.txt; 5,086 text lines, 2,173 parsed source-correlated instruction rows.
- SASS export: medium_d64-tq64-tk64-lat0/source-tileir-sass.txt; 4,948 text lines, 3,265 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (312), mul.rn.f32 (294), add.rn.f32 (216), shl.b32 (117), max.f32 (108), ex2.approx.ftz.f32 (102), fma.rm.f32 (102), sub.rn.f32 (102), add.s32 (94), selp.f32 (94).
- Frequent SASS mnemonics: ffma (377), fadd (318), fmul (294), mov (217), hmma.16816.f32 (192), shf.l.u32 (117), iadd3 (115), lop3.lut (109), fmnmx (108), ffma.rm (103).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | medium_d64-tq64-tk64-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 10.56 Âµs, grid (1, 8, 4), compute 13.89%, DRAM 30.42%. | 3d588a4d2966 |
| 2 | Details scroll 01 | medium_d64-tq64-tk64-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | c0e3d482bd9a |
| 3 | Details scroll 02 | medium_d64-tq64-tk64-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | 9952aa7f8ae0 |
| 4 | Details scroll 03 | medium_d64-tq64-tk64-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | 5521715e16a9 |
| 5 | Details scroll 04 | medium_d64-tq64-tk64-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | eef45dd691b9 |
| 6 | Details scroll 05 | medium_d64-tq64-tk64-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | 7edc8bfeefed |
| 7 | Details scroll 06 | medium_d64-tq64-tk64-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | 7499f3ce086e |
| 8 | Details scroll 07 | medium_d64-tq64-tk64-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | 4e898c32c247 |
| 9 | Details scroll 08 | medium_d64-tq64-tk64-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | 2d36a2dfa173 |
| 10 | Details scroll 09 | medium_d64-tq64-tk64-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 80.84%, achieved occupancy 10.35%, shared memory 49.15 KiB. | 615f17a162c8 |
| 11 | Source scroll 01 | medium_d64-tq64-tk64-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 392cf2e99a79 |
| 12 | Source scroll 02 | medium_d64-tq64-tk64-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | d2aa5be0e94c |
| 13 | Source scroll 03 | medium_d64-tq64-tk64-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | e136ba739d3f |
| 14 | Source scroll 04 | medium_d64-tq64-tk64-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | c1777581d9d5 |
| 15 | Source scroll 05 | medium_d64-tq64-tk64-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | db7c1ac8a2f3 |
| 16 | Source high-level tail | medium_d64-tq64-tk64-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | 7d7298863ab3 |
| 17 | Source high-level head | medium_d64-tq64-tk64-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | 246d85acd16c |
| 18 | SASS at selected line 91 | medium_d64-tq64-tk64-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | a34e62111b94 |
| 19 | Source left code horizontal | medium_d64-tq64-tk64-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | f1c2044fd84d |
| 20 | Source left metrics horizontal | medium_d64-tq64-tk64-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | 4b039914be8a |
| 21 | Source right code horizontal attempt | medium_d64-tq64-tk64-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | bfbf199e4936 |
| 22 | Source right metrics horizontal attempt | medium_d64-tq64-tk64-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | bfbf199e4936 |
| 23 | Source view picker | medium_d64-tq64-tk64-lat0-source-view-picker.png | Right-side view selector state. | 1a689b79a3af |
| 24 | Source view with PTX selected | medium_d64-tq64-tk64-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | cc01a3039759 |
| 25 | PTX at selected line 91 | medium_d64-tq64-tk64-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | cc01a3039759 |
| 26 | Initial PTX horizontal code view | medium_d64-tq64-tk64-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | be26dd66a872 |
| 27 | Initial PTX horizontal metrics view | medium_d64-tq64-tk64-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | 4ee54bd7f219 |
| 28 | Bottom tab: Instructions & Dependencies | medium_d64-tq64-tk64-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 78d2c38af0dc |
| 29 | Bottom tab: Inline Functions | medium_d64-tq64-tk64-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 4dd671ed1b6f |
| 30 | Bottom tab: Source Markers | medium_d64-tq64-tk64-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | 03826c299d8e |
| 31 | Bottom tab: Statistics | medium_d64-tq64-tk64-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 8817fb13aafd |
| 32 | Raw metrics | medium_d64-tq64-tk64-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 7cc5eaa80e27 |
| 33 | PTX supplement: left code | medium_d64-tq64-tk64-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | b7496306b007 |
| 34 | PTX supplement: left metrics | medium_d64-tq64-tk64-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | a3aed4ac5b61 |
| 35 | PTX supplement: right code | medium_d64-tq64-tk64-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | a3aed4ac5b61 |
| 36 | PTX supplement: right metrics | medium_d64-tq64-tk64-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | a3aed4ac5b61 |

### Integrity caveat
- 32 primary screenshots plus 4 supplemental PTX screenshots = 36 named views for this profile.
- All 36 PNGs passed signature and 1936Ã—1048 dimension checks.
- SASS right-code/right-metrics are byte-identical. PTX right-code/right-metrics are also byte-identical. The right-side horizontal movement is therefore not demonstrated; the capture inventory records this instead of claiming success.
- Left code and left metrics are distinct. No assertion of successful right-pane horizontal displacement is made.

## 7. Conclusion
- The NCU report confirms this exact resource, occupancy, scheduler and memory signature on the RTX 4060.
- NCU replay duration and rule-engine Estimated Speedup are not normal performance benchmark results.
- Next optimization step: change only one of TQ, TK or load-latency; run correctness; then conduct a repeated, order-rotated unprofiled CUDA-event benchmark and a matched NCU profile.

[Back to the matrix index](../README.md)
