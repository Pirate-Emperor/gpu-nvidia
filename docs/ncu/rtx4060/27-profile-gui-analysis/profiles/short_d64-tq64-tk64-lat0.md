# short_d64-tq64-tk64-lat0

**Workload:** short_d64 | **TQÃ—TK:** 64Ã—64 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report short_d64-tq64-tk64-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 24.66 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 5.22 Âµs (0.005 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 1, 1) / (128, 1, 1) | 1 CTAs; 128 threads. |
| Compute / memory / DRAM throughput | 0.49% / 5.89% / 5.89% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 10.40% / 2.59% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 15.36 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 60.81% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 2.59% / 0.27% | Neither is evidence of saturated DRAM. |
| Registers / thread | 254 | Per-thread register demand. |
| Static shared memory / CTA | 49.15 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 16.67% / 8.25% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 3.96 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 2 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 85.69% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.14 / 0.14 / 1 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 7.02 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 85.69% and Eligible Warps Per Scheduler is 0.14. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 49.15 KiB static shared memory per CTA; the shared-memory block limit is 2 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 5.89% and bandwidth 15.36 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| LaunchConfiguration | 95.83% | The grid for this launch is configured to execute only 1 block, which is less than the 24 multiprocessors used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently with other workloads, consider reducing the block size to have at least one block per multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more details on launch configurations. |
| IssueSlotUtilization | 85.69% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 7.0 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, but only an average of 0.14 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons on the Warp State Statistics and Source Counters sections. |
| TheoreticalOccupancy | 83.33% | The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory. |
| WorkloadImbalance | 6.07% | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 27.16% above the average, while the minimum instance value is 21.66% below the average. |

## 5. PTX/SASS evidence
- PTX export: short_d64-tq64-tk64-lat0/source-tileir-ptx.txt; 5,140 text lines, 2,185 parsed source-correlated instruction rows.
- SASS export: short_d64-tq64-tk64-lat0/source-tileir-sass.txt; 4,903 text lines, 3,248 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (312), mul.rn.f32 (294), add.rn.f32 (216), shl.b32 (117), max.f32 (108), ex2.approx.ftz.f32 (102), fma.rm.f32 (102), sub.rn.f32 (102), add.s32 (94), selp.f32 (94).
- Frequent SASS mnemonics: ffma (377), fadd (318), fmul (294), mov (209), hmma.16816.f32 (192), shf.l.u32 (117), iadd3 (115), lop3.lut (109), fmnmx (108), ffma.rm (103).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | short_d64-tq64-tk64-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 5.22 Âµs, grid (1, 1, 1), compute 0.49%, DRAM 5.89%. | 6d570b25f82c |
| 2 | Details scroll 01 | short_d64-tq64-tk64-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | d761fcc2f0d3 |
| 3 | Details scroll 02 | short_d64-tq64-tk64-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | 108c816d201b |
| 4 | Details scroll 03 | short_d64-tq64-tk64-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | 6c8d9ec2cfe1 |
| 5 | Details scroll 04 | short_d64-tq64-tk64-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | e0b49ab7215e |
| 6 | Details scroll 05 | short_d64-tq64-tk64-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | 9f0e246d78c1 |
| 7 | Details scroll 06 | short_d64-tq64-tk64-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | 5747d7f8237e |
| 8 | Details scroll 07 | short_d64-tq64-tk64-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | 48c5521f3cad |
| 9 | Details scroll 08 | short_d64-tq64-tk64-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | 686f4174f2c8 |
| 10 | Details scroll 09 | short_d64-tq64-tk64-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.25%, shared memory 49.15 KiB. | b76d14f24bb8 |
| 11 | Source scroll 01 | short_d64-tq64-tk64-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | e5df77fdb703 |
| 12 | Source scroll 02 | short_d64-tq64-tk64-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 1d5467602e76 |
| 13 | Source scroll 03 | short_d64-tq64-tk64-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 3ae85016d063 |
| 14 | Source scroll 04 | short_d64-tq64-tk64-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 8d3c2aaf0463 |
| 15 | Source scroll 05 | short_d64-tq64-tk64-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 9b784a9ae344 |
| 16 | Source high-level tail | short_d64-tq64-tk64-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | 27fb21d848b2 |
| 17 | Source high-level head | short_d64-tq64-tk64-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | a189063fa390 |
| 18 | SASS at selected line 91 | short_d64-tq64-tk64-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | 61f8df221caf |
| 19 | Source left code horizontal | short_d64-tq64-tk64-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | 02c9a3fce0f8 |
| 20 | Source left metrics horizontal | short_d64-tq64-tk64-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | 483efd68b88c |
| 21 | Source right code horizontal attempt | short_d64-tq64-tk64-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 09535bad84f1 |
| 22 | Source right metrics horizontal attempt | short_d64-tq64-tk64-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 09535bad84f1 |
| 23 | Source view picker | short_d64-tq64-tk64-lat0-source-view-picker.png | Right-side view selector state. | 7a180fcf76d0 |
| 24 | Source view with PTX selected | short_d64-tq64-tk64-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 955ef1111161 |
| 25 | PTX at selected line 91 | short_d64-tq64-tk64-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 955ef1111161 |
| 26 | Initial PTX horizontal code view | short_d64-tq64-tk64-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 9a0ca9a45ef4 |
| 27 | Initial PTX horizontal metrics view | short_d64-tq64-tk64-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | a7d4af4559a4 |
| 28 | Bottom tab: Instructions & Dependencies | short_d64-tq64-tk64-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | ccc332dbe653 |
| 29 | Bottom tab: Inline Functions | short_d64-tq64-tk64-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | f58dcf0017dd |
| 30 | Bottom tab: Source Markers | short_d64-tq64-tk64-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | 370ab596bacc |
| 31 | Bottom tab: Statistics | short_d64-tq64-tk64-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | c7a82ab7b08d |
| 32 | Raw metrics | short_d64-tq64-tk64-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 2df2732b8a15 |
| 33 | PTX supplement: left code | short_d64-tq64-tk64-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 50b1cc0e8b6d |
| 34 | PTX supplement: left metrics | short_d64-tq64-tk64-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 2a306ed0cbe7 |
| 35 | PTX supplement: right code | short_d64-tq64-tk64-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | 2a306ed0cbe7 |
| 36 | PTX supplement: right metrics | short_d64-tq64-tk64-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | 2a306ed0cbe7 |

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
