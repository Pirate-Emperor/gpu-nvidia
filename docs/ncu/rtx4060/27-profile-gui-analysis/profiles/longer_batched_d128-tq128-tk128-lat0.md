# longer_batched_d128-tq128-tk128-lat0

**Workload:** longer_batched_d128 | **TQÃ—TK:** 128Ã—128 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report longer_batched_d128-tq128-tk128-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 23.51 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 1,370 Âµs (1.370 ms) | Original metric unit: ms; instrumented value, not normal CUDA-event latency. |
| Grid / block | (2, 8, 16) / (128, 1, 1) | 256 CTAs; 32,768 threads. |
| Compute / memory / DRAM throughput | 21.65% / 43.69% / 9.79% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 34.68% / 43.69% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 25.83 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 96.91% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 39.73% / 16.51% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 98.30 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.35% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4.01 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 88.09% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.12 / 0.12 / 1 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 8.38 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 88.09% and Eligible Warps Per Scheduler is 0.12. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 98.30 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 9.79% and bandwidth 25.83 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 56.31% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 8.4 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, which already limits the scheduler to less than a warp per instruction. |
| TheoreticalOccupancy | 56.31% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| WorkloadImbalance | 6.55% | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 7.04% above the average, while the minimum instance value is 8.19% below the average. |
| WorkloadImbalance | 6.25% | One or more SMs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 6.75% above the average, while the minimum instance value is 8.24% below the average. |
| WorkloadImbalance | 6.25% | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Additionally, other L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 6.75% above the average, while the minimum instance value is 8.24% below the average. |

## 5. PTX/SASS evidence
- PTX export: longer_batched_d128-tq128-tk128-lat0/source-tileir-ptx.txt; 7,038 text lines, 2,852 parsed source-correlated instruction rows.
- SASS export: longer_batched_d128-tq128-tk128-lat0/source-tileir-sass.txt; 8,616 text lines, 6,852 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (400), mul.rn.f32 (388), add.rn.f32 (272), shl.b32 (142), max.f32 (136), ex2.approx.ftz.f32 (132), fma.rm.f32 (132), sub.rn.f32 (132), mov.b32 (129), cvt.rn.f16x2.f32 (128).
- Frequent SASS mnemonics: ffma (915), mov (808), hmma.16816.f32 (512), fadd (404), fmul (388), stl (296), ldl.lu (288), ldl (247), fmnmx (136), ffma.rm (133).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | longer_batched_d128-tq128-tk128-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 1,370 Âµs, grid (2, 8, 16), compute 21.65%, DRAM 9.79%. | fd32d4b20667 |
| 2 | Details scroll 01 | longer_batched_d128-tq128-tk128-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 01ee33c5db6d |
| 3 | Details scroll 02 | longer_batched_d128-tq128-tk128-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 5ac3638b13e0 |
| 4 | Details scroll 03 | longer_batched_d128-tq128-tk128-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 56d7a2615a17 |
| 5 | Details scroll 04 | longer_batched_d128-tq128-tk128-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 734b2faaae2c |
| 6 | Details scroll 05 | longer_batched_d128-tq128-tk128-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | d143436f91f5 |
| 7 | Details scroll 06 | longer_batched_d128-tq128-tk128-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | c82d91ec90db |
| 8 | Details scroll 07 | longer_batched_d128-tq128-tk128-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | bfd025f90559 |
| 9 | Details scroll 08 | longer_batched_d128-tq128-tk128-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 3922a6d1e19c |
| 10 | Details scroll 09 | longer_batched_d128-tq128-tk128-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.09%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 6918bd7d2faa |
| 11 | Source scroll 01 | longer_batched_d128-tq128-tk128-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 7de6a24f5dde |
| 12 | Source scroll 02 | longer_batched_d128-tq128-tk128-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 75b564d9a3dc |
| 13 | Source scroll 03 | longer_batched_d128-tq128-tk128-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | f25b5bfae777 |
| 14 | Source scroll 04 | longer_batched_d128-tq128-tk128-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 23688e705d27 |
| 15 | Source scroll 05 | longer_batched_d128-tq128-tk128-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 665ce1e4ec14 |
| 16 | Source high-level tail | longer_batched_d128-tq128-tk128-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | 610f792eb8f3 |
| 17 | Source high-level head | longer_batched_d128-tq128-tk128-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | da71c3b5463f |
| 18 | SASS at selected line 91 | longer_batched_d128-tq128-tk128-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | 9604946e5eed |
| 19 | Source left code horizontal | longer_batched_d128-tq128-tk128-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | 2ef1377ad1e7 |
| 20 | Source left metrics horizontal | longer_batched_d128-tq128-tk128-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | 32d73eb6610f |
| 21 | Source right code horizontal attempt | longer_batched_d128-tq128-tk128-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 0540f36d9d24 |
| 22 | Source right metrics horizontal attempt | longer_batched_d128-tq128-tk128-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 0540f36d9d24 |
| 23 | Source view picker | longer_batched_d128-tq128-tk128-lat0-source-view-picker.png | Right-side view selector state. | 5456c7ea4276 |
| 24 | Source view with PTX selected | longer_batched_d128-tq128-tk128-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 570e5dc8ece4 |
| 25 | PTX at selected line 91 | longer_batched_d128-tq128-tk128-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 570e5dc8ece4 |
| 26 | Initial PTX horizontal code view | longer_batched_d128-tq128-tk128-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 33f0d3b954d6 |
| 27 | Initial PTX horizontal metrics view | longer_batched_d128-tq128-tk128-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | 2d54741552a6 |
| 28 | Bottom tab: Instructions & Dependencies | longer_batched_d128-tq128-tk128-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 4f11de0f2885 |
| 29 | Bottom tab: Inline Functions | longer_batched_d128-tq128-tk128-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 02bb43469ce0 |
| 30 | Bottom tab: Source Markers | longer_batched_d128-tq128-tk128-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | c43ec38f6b8e |
| 31 | Bottom tab: Statistics | longer_batched_d128-tq128-tk128-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | d76601a74e99 |
| 32 | Raw metrics | longer_batched_d128-tq128-tk128-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 8d94719f4757 |
| 33 | PTX supplement: left code | longer_batched_d128-tq128-tk128-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | d5eb58bae467 |
| 34 | PTX supplement: left metrics | longer_batched_d128-tq128-tk128-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | b825ed519d64 |
| 35 | PTX supplement: right code | longer_batched_d128-tq128-tk128-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | b825ed519d64 |
| 36 | PTX supplement: right metrics | longer_batched_d128-tq128-tk128-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | b825ed519d64 |

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
