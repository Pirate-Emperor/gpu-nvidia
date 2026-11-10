# medium_d128-tq64-tk64-lat10

**Workload:** medium_d128 | **TQÃ—TK:** 64Ã—64 | **load latency:** 10 | **status:** PASS

## 1. Provenance
- Native report medium_d128-tq64-tk64-lat10.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 23.17 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 47.30 Âµs (0.047 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 8, 8) / (128, 1, 1) | 64 CTAs; 8,192 threads. |
| Compute / memory / DRAM throughput | 21.04% / 25.82% / 25.82% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 20.19% / 16.35% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 68.07 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 72.66% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 16.52% / 8.92% | Neither is evidence of saturated DRAM. |
| Registers / thread | 238 | Per-thread register demand. |
| Static shared memory / CTA | 98.30 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.37% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4.02 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 84.03% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.16 / 0.16 / 1.01 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 6.31 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 84.03% and Eligible Warps Per Scheduler is 0.16. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 98.30 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 25.82% and bandwidth 68.07 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 74.18% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 6.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.01 active warps per scheduler, but only an average of 0.16 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. |
| TheoreticalOccupancy | 74.18% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| CPIStall | 36.11% | On average, each warp of this workload spends 2.3 cycles being stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. When this reason dominates, the kernel is usually compute-pipeline-bound; otherwise, increasing active warps can help hide latency. This stall type represents about 36.1% of the total average of 6.3 cycles between issuing two instructions. |
| LaunchConfiguration | 33.33% | A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in 2 full waves and a partial wave of 16 thread blocks. Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for up to 33.3% of the total runtime of this kernel. Try launching a grid with no partial wave. The overall impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more details on launch configurations. |
| WorkloadImbalance | 13.13% | One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 16.04% above the average, while the minimum instance value is 25.73% below the average. |
| WorkloadImbalance | 13.13% | One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 16.04% above the average, while the minimum instance value is 25.73% below the average. |
| WorkloadImbalance | 13.08% | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 16.02% above the average, while the minimum instance value is 25.05% below the average. |

## 5. PTX/SASS evidence
- PTX export: medium_d128-tq64-tk64-lat10/source-tileir-ptx.txt; 6,514 text lines, 2,555 parsed source-correlated instruction rows.
- SASS export: medium_d128-tq64-tk64-lat10/source-tileir-sass.txt; 6,535 text lines, 4,522 parsed disassembly rows.
- Frequent PTX mnemonics: mul.rn.f32 (390), fma.rn.f32 (312), add.rn.f32 (216), add.s32 (127), shl.b32 (125), max.f32 (108), ex2.approx.ftz.f32 (102), fma.rm.f32 (102), sub.rn.f32 (102), mov.b32 (98).
- Frequent SASS mnemonics: ffma (537), fmul (390), hmma.16816.f32 (384), fadd (318), mov (296), iadd3 (148), lop3.lut (139), prmt (128), fmnmx (108), ffma.rm (103).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | medium_d128-tq64-tk64-lat10-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 47.30 Âµs, grid (1, 8, 8), compute 21.04%, DRAM 25.82%. | e95c79c7c9cf |
| 2 | Details scroll 01 | medium_d128-tq64-tk64-lat10-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | b8a7b4cdf56d |
| 3 | Details scroll 02 | medium_d128-tq64-tk64-lat10-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | c7d14db5af1a |
| 4 | Details scroll 03 | medium_d128-tq64-tk64-lat10-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | e132b7d24e90 |
| 5 | Details scroll 04 | medium_d128-tq64-tk64-lat10-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | 5697d547e961 |
| 6 | Details scroll 05 | medium_d128-tq64-tk64-lat10-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | 19494395a46a |
| 7 | Details scroll 06 | medium_d128-tq64-tk64-lat10-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | da8b814c1f4c |
| 8 | Details scroll 07 | medium_d128-tq64-tk64-lat10-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | 8f77b5efdfe2 |
| 9 | Details scroll 08 | medium_d128-tq64-tk64-lat10-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | 6dcebba52992 |
| 10 | Details scroll 09 | medium_d128-tq64-tk64-lat10-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 84.03%, achieved occupancy 8.37%, shared memory 98.30 KiB. | 6818336b5b46 |
| 11 | Source scroll 01 | medium_d128-tq64-tk64-lat10-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 555fd97757f7 |
| 12 | Source scroll 02 | medium_d128-tq64-tk64-lat10-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 64d3817847dd |
| 13 | Source scroll 03 | medium_d128-tq64-tk64-lat10-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | a855ac319b4a |
| 14 | Source scroll 04 | medium_d128-tq64-tk64-lat10-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | d73629614b7c |
| 15 | Source scroll 05 | medium_d128-tq64-tk64-lat10-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 0100e5c4b3a1 |
| 16 | Source high-level tail | medium_d128-tq64-tk64-lat10-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | 72e9323eb8a1 |
| 17 | Source high-level head | medium_d128-tq64-tk64-lat10-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | 7271154c8c17 |
| 18 | SASS at selected line 91 | medium_d128-tq64-tk64-lat10-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | 262e8118ca47 |
| 19 | Source left code horizontal | medium_d128-tq64-tk64-lat10-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | f7181ef61ca3 |
| 20 | Source left metrics horizontal | medium_d128-tq64-tk64-lat10-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | f9f23187b6e2 |
| 21 | Source right code horizontal attempt | medium_d128-tq64-tk64-lat10-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 24efe5cfc9b6 |
| 22 | Source right metrics horizontal attempt | medium_d128-tq64-tk64-lat10-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 24efe5cfc9b6 |
| 23 | Source view picker | medium_d128-tq64-tk64-lat10-source-view-picker.png | Right-side view selector state. | ec147d2269f3 |
| 24 | Source view with PTX selected | medium_d128-tq64-tk64-lat10-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | d883fdb519f7 |
| 25 | PTX at selected line 91 | medium_d128-tq64-tk64-lat10-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | d883fdb519f7 |
| 26 | Initial PTX horizontal code view | medium_d128-tq64-tk64-lat10-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 02dd04f4ff56 |
| 27 | Initial PTX horizontal metrics view | medium_d128-tq64-tk64-lat10-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | 8dd592441474 |
| 28 | Bottom tab: Instructions & Dependencies | medium_d128-tq64-tk64-lat10-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 902c02af7291 |
| 29 | Bottom tab: Inline Functions | medium_d128-tq64-tk64-lat10-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 8efe4f6e6543 |
| 30 | Bottom tab: Source Markers | medium_d128-tq64-tk64-lat10-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | 020e7c68d74c |
| 31 | Bottom tab: Statistics | medium_d128-tq64-tk64-lat10-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 87a5d894c6b9 |
| 32 | Raw metrics | medium_d128-tq64-tk64-lat10-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 269f54dee766 |
| 33 | PTX supplement: left code | medium_d128-tq64-tk64-lat10-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 8b420d4c13a2 |
| 34 | PTX supplement: left metrics | medium_d128-tq64-tk64-lat10-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 2723a3e17e6e |
| 35 | PTX supplement: right code | medium_d128-tq64-tk64-lat10-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | 8ec2a7840e63 |
| 36 | PTX supplement: right metrics | medium_d128-tq64-tk64-lat10-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | 8ec2a7840e63 |

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
