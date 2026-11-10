# medium_d128-tq128-tk128-lat0

**Workload:** medium_d128 | **TQÃ—TK:** 128Ã—128 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report medium_d128-tq128-tk128-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 24.96 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 106.56 Âµs (0.107 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 8, 4) / (128, 1, 1) | 32 CTAs; 4,096 threads. |
| Compute / memory / DRAM throughput | 10.31% / 24.12% / 11.60% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 19.14% / 24.12% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 30.59 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 90.80% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 21.76% / 9.21% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 98.30 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.29% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 3.98 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 90.78% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.09 / 0.09 / 1.01 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 10.96 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 90.78% and Eligible Warps Per Scheduler is 0.09. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 98.30 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 11.60% and bandwidth 30.59 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 75.88% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 10.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.01 active warps per scheduler, but only an average of 0.09 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. |
| TheoreticalOccupancy | 75.88% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| LaunchConfiguration | 50% | A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in 1 full waves and a partial wave of 8 thread blocks. Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for up to 50.0% of the total runtime of this kernel. Try launching a grid with no partial wave. The overall impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more details on launch configurations. |
| CPIStall | 40.66% | On average, each warp of this workload spends 4.5 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory. Stall samples are reported on the consumer instruction waiting for the result, not necessarily on the producer. When this reason dominates, the kernel is memory-latency-bound, which is not necessarily memory-bandwidth-bound; prefer latency-hiding changes such as ILP, unrolling, or pipelining when bandwidth is not saturated. This stall type represents about 40.7% of the total average of 11.0 cycles between issuing two instructions. |
| WorkloadImbalance | 21.32% | One or more SMs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 31.44% above the average, while the minimum instance value is 31.71% below the average. |
| WorkloadImbalance | 21.32% | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Additionally, other L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 31.44% above the average, while the minimum instance value is 31.71% below the average. |
| WorkloadImbalance | 21.22% | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 31.80% above the average, while the minimum instance value is 31.94% below the average. |

## 5. PTX/SASS evidence
- PTX export: medium_d128-tq128-tk128-lat0/source-tileir-ptx.txt; 6,992 text lines, 2,825 parsed source-correlated instruction rows.
- SASS export: medium_d128-tq128-tk128-lat0/source-tileir-sass.txt; 8,597 text lines, 6,866 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (400), mul.rn.f32 (388), add.rn.f32 (272), shl.b32 (142), max.f32 (136), ex2.approx.ftz.f32 (132), fma.rm.f32 (132), sub.rn.f32 (132), mov.b32 (129), cvt.rn.f16x2.f32 (128).
- Frequent SASS mnemonics: ffma (915), mov (800), hmma.16816.f32 (512), fadd (404), fmul (388), stl (296), ldl.lu (288), ldl (247), fmnmx (136), ffma.rm (133).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | medium_d128-tq128-tk128-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 106.56 Âµs, grid (1, 8, 4), compute 10.31%, DRAM 11.60%. | ead2dbd24c91 |
| 2 | Details scroll 01 | medium_d128-tq128-tk128-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | af093baf3972 |
| 3 | Details scroll 02 | medium_d128-tq128-tk128-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | 6f9dcc140a9f |
| 4 | Details scroll 03 | medium_d128-tq128-tk128-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | 7049b19d93f4 |
| 5 | Details scroll 04 | medium_d128-tq128-tk128-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | b4ad06bfdbe2 |
| 6 | Details scroll 05 | medium_d128-tq128-tk128-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | 0332ca6bc966 |
| 7 | Details scroll 06 | medium_d128-tq128-tk128-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | 2c2e775a38fd |
| 8 | Details scroll 07 | medium_d128-tq128-tk128-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | 36436c517a89 |
| 9 | Details scroll 08 | medium_d128-tq128-tk128-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | 6e6c66fb4921 |
| 10 | Details scroll 09 | medium_d128-tq128-tk128-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 90.78%, achieved occupancy 8.29%, shared memory 98.30 KiB. | 92f7dff4e381 |
| 11 | Source scroll 01 | medium_d128-tq128-tk128-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | d7252381d5bb |
| 12 | Source scroll 02 | medium_d128-tq128-tk128-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 5f36938726fa |
| 13 | Source scroll 03 | medium_d128-tq128-tk128-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | b06e64044b49 |
| 14 | Source scroll 04 | medium_d128-tq128-tk128-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | e8b11be4326c |
| 15 | Source scroll 05 | medium_d128-tq128-tk128-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | d444c5f63aab |
| 16 | Source high-level tail | medium_d128-tq128-tk128-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | 1ba0801cb0d1 |
| 17 | Source high-level head | medium_d128-tq128-tk128-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | b0a8c65ad8bb |
| 18 | SASS at selected line 91 | medium_d128-tq128-tk128-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | a5277067daab |
| 19 | Source left code horizontal | medium_d128-tq128-tk128-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | da9c4b0302d7 |
| 20 | Source left metrics horizontal | medium_d128-tq128-tk128-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | 491454e188e6 |
| 21 | Source right code horizontal attempt | medium_d128-tq128-tk128-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 244bcb19dd3b |
| 22 | Source right metrics horizontal attempt | medium_d128-tq128-tk128-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 244bcb19dd3b |
| 23 | Source view picker | medium_d128-tq128-tk128-lat0-source-view-picker.png | Right-side view selector state. | 19983640866d |
| 24 | Source view with PTX selected | medium_d128-tq128-tk128-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 327da6048d60 |
| 25 | PTX at selected line 91 | medium_d128-tq128-tk128-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 327da6048d60 |
| 26 | Initial PTX horizontal code view | medium_d128-tq128-tk128-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | ffa973c73c37 |
| 27 | Initial PTX horizontal metrics view | medium_d128-tq128-tk128-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | dd52a7b84942 |
| 28 | Bottom tab: Instructions & Dependencies | medium_d128-tq128-tk128-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | c79e39c6e1fb |
| 29 | Bottom tab: Inline Functions | medium_d128-tq128-tk128-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | a2019939dfc7 |
| 30 | Bottom tab: Source Markers | medium_d128-tq128-tk128-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | d42a83c85b6f |
| 31 | Bottom tab: Statistics | medium_d128-tq128-tk128-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 4099f4f2fc9b |
| 32 | Raw metrics | medium_d128-tq128-tk128-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 78f02dc8e963 |
| 33 | PTX supplement: left code | medium_d128-tq128-tk128-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 5d4bb44b9741 |
| 34 | PTX supplement: left metrics | medium_d128-tq128-tk128-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 1468f15489d3 |
| 35 | PTX supplement: right code | medium_d128-tq128-tk128-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | 6e3732d5d56d |
| 36 | PTX supplement: right metrics | medium_d128-tq128-tk128-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | 6e3732d5d56d |

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
