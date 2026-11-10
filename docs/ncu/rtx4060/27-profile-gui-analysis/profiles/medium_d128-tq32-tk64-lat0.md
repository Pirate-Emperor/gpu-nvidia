# medium_d128-tq32-tk64-lat0

**Workload:** medium_d128 | **TQÃ—TK:** 32Ã—64 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report medium_d128-tq32-tk64-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 24.66 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 63.07 Âµs (0.063 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 8, 16) / (128, 1, 1) | 128 CTAs; 16,384 threads. |
| Compute / memory / DRAM throughput | 15.77% / 21.98% / 19.33% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 19.10% / 21.98% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 50.93 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 84.84% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 21.98% / 12.76% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 78.08 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.37% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4.02 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 82.74% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.17 / 0.17 / 1.01 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 5.83 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 82.74% and Eligible Warps Per Scheduler is 0.17. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 78.08 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 19.33% and bandwidth 50.93 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 78.02% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 5.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.01 active warps per scheduler, but only an average of 0.17 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. |
| TheoreticalOccupancy | 78.02% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| WorkloadImbalance | 9.48% | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 10.93% above the average, while the minimum instance value is 10.76% below the average. |
| WorkloadImbalance | 9.44% | One or more SMs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 10.84% above the average, while the minimum instance value is 10.77% below the average. |
| WorkloadImbalance | 9.44% | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Additionally, other L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 10.84% above the average, while the minimum instance value is 10.77% below the average. |

## 5. PTX/SASS evidence
- PTX export: medium_d128-tq32-tk64-lat0/source-tileir-ptx.txt; 6,149 text lines, 2,556 parsed source-correlated instruction rows.
- SASS export: medium_d128-tq32-tk64-lat0/source-tileir-sass.txt; 5,899 text lines, 4,061 parsed disassembly rows.
- Frequent PTX mnemonics: mov.b32 (310), mul.rn.f32 (272), fma.rn.f32 (256), add.rn.f32 (176), add.s32 (135), shl.b32 (104), max.f32 (96), selp.b32 (88), xor.b32 (82), ex2.approx.ftz.f32 (80).
- Frequent SASS mnemonics: ffma (343), prmt (304), fmul (272), fadd (256), hmma.16816.f32 (256), imad.mov.u32 (192), lop3.lut (190), iadd3 (140), fmnmx (96), imad.shl.u32 (95).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | medium_d128-tq32-tk64-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 63.07 Âµs, grid (1, 8, 16), compute 15.77%, DRAM 19.33%. | d791d1024c26 |
| 2 | Details scroll 01 | medium_d128-tq32-tk64-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | 6ab3fc84d043 |
| 3 | Details scroll 02 | medium_d128-tq32-tk64-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | 9b6036b49149 |
| 4 | Details scroll 03 | medium_d128-tq32-tk64-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | a5e0c9b52c3d |
| 5 | Details scroll 04 | medium_d128-tq32-tk64-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | 24765e059a74 |
| 6 | Details scroll 05 | medium_d128-tq32-tk64-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | 31c3a8e5c291 |
| 7 | Details scroll 06 | medium_d128-tq32-tk64-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | 23deac77773d |
| 8 | Details scroll 07 | medium_d128-tq32-tk64-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | 177a3abfc043 |
| 9 | Details scroll 08 | medium_d128-tq32-tk64-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | ef3a91427709 |
| 10 | Details scroll 09 | medium_d128-tq32-tk64-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 82.74%, achieved occupancy 8.37%, shared memory 78.08 KiB. | 34cef649d2ba |
| 11 | Source scroll 01 | medium_d128-tq32-tk64-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | f73e50825f7c |
| 12 | Source scroll 02 | medium_d128-tq32-tk64-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | f94e292e2f29 |
| 13 | Source scroll 03 | medium_d128-tq32-tk64-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | e5b44e30aa50 |
| 14 | Source scroll 04 | medium_d128-tq32-tk64-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 5bf9aa874d9c |
| 15 | Source scroll 05 | medium_d128-tq32-tk64-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 3f1bd5376373 |
| 16 | Source high-level tail | medium_d128-tq32-tk64-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | fa8380de10de |
| 17 | Source high-level head | medium_d128-tq32-tk64-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | 08006a28f28d |
| 18 | SASS at selected line 91 | medium_d128-tq32-tk64-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | f36b12033143 |
| 19 | Source left code horizontal | medium_d128-tq32-tk64-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | e8fe4d9a26dd |
| 20 | Source left metrics horizontal | medium_d128-tq32-tk64-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | c215ecf8790c |
| 21 | Source right code horizontal attempt | medium_d128-tq32-tk64-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 56f066aee0d6 |
| 22 | Source right metrics horizontal attempt | medium_d128-tq32-tk64-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 56f066aee0d6 |
| 23 | Source view picker | medium_d128-tq32-tk64-lat0-source-view-picker.png | Right-side view selector state. | 79bde3cda94d |
| 24 | Source view with PTX selected | medium_d128-tq32-tk64-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 5d7cc9175579 |
| 25 | PTX at selected line 91 | medium_d128-tq32-tk64-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 5d7cc9175579 |
| 26 | Initial PTX horizontal code view | medium_d128-tq32-tk64-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | ca1200b7b70f |
| 27 | Initial PTX horizontal metrics view | medium_d128-tq32-tk64-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | cb3f73ad08d6 |
| 28 | Bottom tab: Instructions & Dependencies | medium_d128-tq32-tk64-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 85a7866c0471 |
| 29 | Bottom tab: Inline Functions | medium_d128-tq32-tk64-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 2d1696c77588 |
| 30 | Bottom tab: Source Markers | medium_d128-tq32-tk64-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | c660c0502a67 |
| 31 | Bottom tab: Statistics | medium_d128-tq32-tk64-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 7712db4cced3 |
| 32 | Raw metrics | medium_d128-tq32-tk64-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | f4c429b8176c |
| 33 | PTX supplement: left code | medium_d128-tq32-tk64-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 612320dcbe2d |
| 34 | PTX supplement: left metrics | medium_d128-tq32-tk64-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 13ac1b525ebd |
| 35 | PTX supplement: right code | medium_d128-tq32-tk64-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | cb3de5056e48 |
| 36 | PTX supplement: right metrics | medium_d128-tq32-tk64-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | cb3de5056e48 |

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
