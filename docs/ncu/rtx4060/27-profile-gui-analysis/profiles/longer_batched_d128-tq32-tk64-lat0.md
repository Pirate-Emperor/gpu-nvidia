# longer_batched_d128-tq32-tk64-lat0

**Workload:** longer_batched_d128 | **TQÃ—TK:** 32Ã—64 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report longer_batched_d128-tq32-tk64-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 23.12 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 1,230 Âµs (1.230 ms) | Original metric unit: ms; instrumented value, not normal CUDA-event latency. |
| Grid / block | (2, 8, 64) / (128, 1, 1) | 1,024 CTAs; 131,072 threads. |
| Compute / memory / DRAM throughput | 23.57% / 30.95% / 9.54% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 24.75% / 30.95% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 25.17 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 95.69% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 30.95% / 18.47% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 78.08 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.33% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 79.67% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.20 / 0.20 / 1 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 4.92 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 79.67% and Eligible Warps Per Scheduler is 0.20. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 78.08 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 9.54% and bandwidth 25.17 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 69.05% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 4.9 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, but only an average of 0.20 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. |
| TheoreticalOccupancy | 69.05% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| CPIStall | 32.91% | On average, each warp of this workload spends 1.6 cycles being stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. When this reason dominates, the kernel is usually compute-pipeline-bound; otherwise, increasing active warps can help hide latency. This stall type represents about 32.9% of the total average of 4.9 cycles between issuing two instructions. |

## 5. PTX/SASS evidence
- PTX export: longer_batched_d128-tq32-tk64-lat0/source-tileir-ptx.txt; 6,262 text lines, 2,599 parsed source-correlated instruction rows.
- SASS export: longer_batched_d128-tq32-tk64-lat0/source-tileir-sass.txt; 5,854 text lines, 4,055 parsed disassembly rows.
- Frequent PTX mnemonics: mov.b32 (310), mul.rn.f32 (272), fma.rn.f32 (256), add.rn.f32 (176), add.s32 (135), shl.b32 (108), max.f32 (96), selp.b32 (96), xor.b32 (82), ex2.approx.ftz.f32 (80).
- Frequent SASS mnemonics: ffma (343), prmt (304), fmul (272), fadd (256), hmma.16816.f32 (256), imad.mov.u32 (207), lop3.lut (191), iadd3 (137), imad.shl.u32 (102), sel (98).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | longer_batched_d128-tq32-tk64-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 1,230 Âµs, grid (2, 8, 64), compute 23.57%, DRAM 9.54%. | f19bfef9269d |
| 2 | Details scroll 01 | longer_batched_d128-tq32-tk64-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 428fd3cd5661 |
| 3 | Details scroll 02 | longer_batched_d128-tq32-tk64-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 35f69ae8eead |
| 4 | Details scroll 03 | longer_batched_d128-tq32-tk64-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 014be9323af3 |
| 5 | Details scroll 04 | longer_batched_d128-tq32-tk64-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 9a63526491b6 |
| 6 | Details scroll 05 | longer_batched_d128-tq32-tk64-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | ade91c9bf749 |
| 7 | Details scroll 06 | longer_batched_d128-tq32-tk64-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 3d707da2b1c5 |
| 8 | Details scroll 07 | longer_batched_d128-tq32-tk64-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | adaf2956fe1c |
| 9 | Details scroll 08 | longer_batched_d128-tq32-tk64-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 52196ce16841 |
| 10 | Details scroll 09 | longer_batched_d128-tq32-tk64-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 79.67%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 1980c92d8be8 |
| 11 | Source scroll 01 | longer_batched_d128-tq32-tk64-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 295e1eb0a1ec |
| 12 | Source scroll 02 | longer_batched_d128-tq32-tk64-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | a63afdf391ac |
| 13 | Source scroll 03 | longer_batched_d128-tq32-tk64-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 68522391a9f9 |
| 14 | Source scroll 04 | longer_batched_d128-tq32-tk64-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 98c4430615c7 |
| 15 | Source scroll 05 | longer_batched_d128-tq32-tk64-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 33249ef26fdb |
| 16 | Source high-level tail | longer_batched_d128-tq32-tk64-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | d4a11e29e553 |
| 17 | Source high-level head | longer_batched_d128-tq32-tk64-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | cf4306ebbb04 |
| 18 | SASS at selected line 91 | longer_batched_d128-tq32-tk64-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | b3c064a6061a |
| 19 | Source left code horizontal | longer_batched_d128-tq32-tk64-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | 2af9277227eb |
| 20 | Source left metrics horizontal | longer_batched_d128-tq32-tk64-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | d35622929f98 |
| 21 | Source right code horizontal attempt | longer_batched_d128-tq32-tk64-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 907a88802bb6 |
| 22 | Source right metrics horizontal attempt | longer_batched_d128-tq32-tk64-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 907a88802bb6 |
| 23 | Source view picker | longer_batched_d128-tq32-tk64-lat0-source-view-picker.png | Right-side view selector state. | a7ecb6646d55 |
| 24 | Source view with PTX selected | longer_batched_d128-tq32-tk64-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 857f16079b77 |
| 25 | PTX at selected line 91 | longer_batched_d128-tq32-tk64-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 857f16079b77 |
| 26 | Initial PTX horizontal code view | longer_batched_d128-tq32-tk64-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 6a50757c60dd |
| 27 | Initial PTX horizontal metrics view | longer_batched_d128-tq32-tk64-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | bd0d9c566335 |
| 28 | Bottom tab: Instructions & Dependencies | longer_batched_d128-tq32-tk64-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 0dde998d4342 |
| 29 | Bottom tab: Inline Functions | longer_batched_d128-tq32-tk64-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 07c392526101 |
| 30 | Bottom tab: Source Markers | longer_batched_d128-tq32-tk64-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | 403daf743dde |
| 31 | Bottom tab: Statistics | longer_batched_d128-tq32-tk64-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 45e42cac1435 |
| 32 | Raw metrics | longer_batched_d128-tq32-tk64-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 089f0f448708 |
| 33 | PTX supplement: left code | longer_batched_d128-tq32-tk64-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 76e12feff225 |
| 34 | PTX supplement: left metrics | longer_batched_d128-tq32-tk64-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 5928a1f2977e |
| 35 | PTX supplement: right code | longer_batched_d128-tq32-tk64-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | 5928a1f2977e |
| 36 | PTX supplement: right metrics | longer_batched_d128-tq32-tk64-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | 5928a1f2977e |

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
