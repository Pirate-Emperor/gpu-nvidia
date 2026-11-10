# medium_d64-tq128-tk64-lat0

**Workload:** medium_d64 | **TQÃ—TK:** 128Ã—64 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report medium_d64-tq128-tk64-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 24.64 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 14.78 Âµs (0.015 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 8, 2) / (128, 1, 1) | 16 CTAs; 2,048 threads. |
| Compute / memory / DRAM throughput | 11.07% / 21.99% / 21.99% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 10.61% / 7.60% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 57.87 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 44.11% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 6.81% / 3.79% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 49.15 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 16.67% / 8.42% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4.04 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 2 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 81.90% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.18 / 0.18 / 0.96 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 5.30 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 81.90% and Eligible Warps Per Scheduler is 0.18. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 49.15 KiB static shared memory per CTA; the shared-memory block limit is 2 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 21.99% and bandwidth 57.87 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 78.01% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 5.5 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 0.96 active warps per scheduler, which already limits the scheduler to less than a warp per instruction. |
| TheoreticalOccupancy | 78.01% | The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory. |
| LaunchConfiguration | 33.33% | The grid for this launch is configured to execute only 16 blocks, which is less than the 24 multiprocessors used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently with other workloads, consider reducing the block size to have at least one block per multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more details on launch configurations. |
| CPIStall | 30.51% | On average, each warp of this workload spends 1.6 cycles being stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. When this reason dominates, the kernel is usually compute-pipeline-bound; otherwise, increasing active warps can help hide latency. This stall type represents about 30.5% of the total average of 5.3 cycles between issuing two instructions. |
| WorkloadImbalance | 23.35% | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 44.01% above the average, while the minimum instance value is 100.00% below the average. |
| WorkloadImbalance | 22.88% | One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 44.32% above the average, while the minimum instance value is 100.00% below the average. |
| WorkloadImbalance | 22.88% | One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 44.32% above the average, while the minimum instance value is 100.00% below the average. |
| WorkloadImbalance | 5.85% | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 14.71% above the average, while the minimum instance value is 4.61% below the average. |

## 5. PTX/SASS evidence
- PTX export: medium_d64-tq128-tk64-lat0/source-tileir-ptx.txt; 6,127 text lines, 2,714 parsed source-correlated instruction rows.
- SASS export: medium_d64-tq128-tk64-lat0/source-tileir-sass.txt; 6,395 text lines, 4,248 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (416), mul.rn.f32 (392), add.rn.f32 (288), shl.b32 (149), max.f32 (144), ex2.approx.ftz.f32 (136), fma.rm.f32 (136), sub.rn.f32 (136), selp.f32 (128), setp.gt.s32 (128).
- Frequent SASS mnemonics: ffma (607), fadd (424), fmul (392), mov (285), hmma.16816.f32 (256), shf.l.u32 (157), fmnmx (144), ffma.rm (137), ffma.sat (136), mufu.ex2 (136).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | medium_d64-tq128-tk64-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 14.78 Âµs, grid (1, 8, 2), compute 11.07%, DRAM 21.99%. | c5371997b58c |
| 2 | Details scroll 01 | medium_d64-tq128-tk64-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | 278be2772e34 |
| 3 | Details scroll 02 | medium_d64-tq128-tk64-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | 6d75cf51b622 |
| 4 | Details scroll 03 | medium_d64-tq128-tk64-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | 60f670d0a51f |
| 5 | Details scroll 04 | medium_d64-tq128-tk64-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | 915803be215c |
| 6 | Details scroll 05 | medium_d64-tq128-tk64-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | 3a0c1ee26958 |
| 7 | Details scroll 06 | medium_d64-tq128-tk64-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | 6922941ddadf |
| 8 | Details scroll 07 | medium_d64-tq128-tk64-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | 12653036fec2 |
| 9 | Details scroll 08 | medium_d64-tq128-tk64-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | b1bc80063561 |
| 10 | Details scroll 09 | medium_d64-tq128-tk64-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.90%, achieved occupancy 8.42%, shared memory 49.15 KiB. | a2768ae0c398 |
| 11 | Source scroll 01 | medium_d64-tq128-tk64-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | f49f5b8a113c |
| 12 | Source scroll 02 | medium_d64-tq128-tk64-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 5eea6d3c903b |
| 13 | Source scroll 03 | medium_d64-tq128-tk64-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | a6cd8d2f7714 |
| 14 | Source scroll 04 | medium_d64-tq128-tk64-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 41f1270d43ae |
| 15 | Source scroll 05 | medium_d64-tq128-tk64-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | c5947a6aec86 |
| 16 | Source high-level tail | medium_d64-tq128-tk64-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | c33f6a7eb616 |
| 17 | Source high-level head | medium_d64-tq128-tk64-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | 80107ffffd32 |
| 18 | SASS at selected line 91 | medium_d64-tq128-tk64-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | 0b9d9f0901ef |
| 19 | Source left code horizontal | medium_d64-tq128-tk64-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | 23eb9b8e5042 |
| 20 | Source left metrics horizontal | medium_d64-tq128-tk64-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | fe475eae21b2 |
| 21 | Source right code horizontal attempt | medium_d64-tq128-tk64-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 6b0e5c212bbc |
| 22 | Source right metrics horizontal attempt | medium_d64-tq128-tk64-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 6b0e5c212bbc |
| 23 | Source view picker | medium_d64-tq128-tk64-lat0-source-view-picker.png | Right-side view selector state. | bccfdd6d3281 |
| 24 | Source view with PTX selected | medium_d64-tq128-tk64-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | eebbffe92ee0 |
| 25 | PTX at selected line 91 | medium_d64-tq128-tk64-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | eebbffe92ee0 |
| 26 | Initial PTX horizontal code view | medium_d64-tq128-tk64-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 87e2836ec59d |
| 27 | Initial PTX horizontal metrics view | medium_d64-tq128-tk64-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | e01fd036860a |
| 28 | Bottom tab: Instructions & Dependencies | medium_d64-tq128-tk64-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | d63e19c88597 |
| 29 | Bottom tab: Inline Functions | medium_d64-tq128-tk64-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 87555ae522d9 |
| 30 | Bottom tab: Source Markers | medium_d64-tq128-tk64-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | ecf955d333ba |
| 31 | Bottom tab: Statistics | medium_d64-tq128-tk64-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 70009b82f8cf |
| 32 | Raw metrics | medium_d64-tq128-tk64-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 30ef2247fe4c |
| 33 | PTX supplement: left code | medium_d64-tq128-tk64-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 44c33391c6a0 |
| 34 | PTX supplement: left metrics | medium_d64-tq128-tk64-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | ebd10563199b |
| 35 | PTX supplement: right code | medium_d64-tq128-tk64-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | ebd10563199b |
| 36 | PTX supplement: right metrics | medium_d64-tq128-tk64-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | ebd10563199b |

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
