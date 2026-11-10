# medium_d64-tq64-tk128-lat0

**Workload:** medium_d64 | **TQÃ—TK:** 64Ã—128 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report medium_d64-tq64-tk128-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 24.46 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 11.81 Âµs (0.012 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 8, 4) / (128, 1, 1) | 32 CTAs; 4,096 threads. |
| Compute / memory / DRAM throughput | 14.35% / 26.75% / 26.75% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 15.91% / 11.17% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 70.16 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 58.01% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 11.63% / 7.19% | Neither is evidence of saturated DRAM. |
| Registers / thread | 248 | Per-thread register demand. |
| Static shared memory / CTA | 49.15 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 16.67% / 10.46% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 5.02 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 2 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 81.34% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.21 / 0.19 / 1.25 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 6.69 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 81.34% and Eligible Warps Per Scheduler is 0.21. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 49.15 KiB static shared memory per CTA; the shared-memory block limit is 2 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 26.75% and bandwidth 70.16 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 73.25% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 5.4 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.25 active warps per scheduler, but only an average of 0.21 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons on the Warp State Statistics and Source Counters sections. |
| TheoreticalOccupancy | 73.25% | The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory. |
| WorkloadImbalance | 15.53% | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 21.22% above the average, while the minimum instance value is 32.02% below the average. |
| WorkloadImbalance | 15.27% | One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 20.89% above the average, while the minimum instance value is 31.91% below the average. |
| WorkloadImbalance | 15.27% | One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 20.89% above the average, while the minimum instance value is 31.91% below the average. |

## 5. PTX/SASS evidence
- PTX export: medium_d64-tq64-tk128-lat0/source-tileir-ptx.txt; 3,572 text lines, 1,467 parsed source-correlated instruction rows.
- SASS export: medium_d64-tq64-tk128-lat0/source-tileir-sass.txt; 3,602 text lines, 2,411 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (200), mul.rn.f32 (162), add.rn.f32 (136), shl.b32 (77), add.s32 (76), max.f32 (68), ex2.approx.ftz.f32 (66), fma.rm.f32 (66), sub.rn.f32 (66), selp.f32 (64).
- Frequent SASS mnemonics: ffma (301), fadd (202), fmul (162), mov (159), hmma.16816.f32 (128), iadd3 (102), lop3.lut (91), fmnmx (68), ffma.rm (67), ffma.sat (66).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | medium_d64-tq64-tk128-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 11.81 Âµs, grid (1, 8, 4), compute 14.35%, DRAM 26.75%. | b8130491751c |
| 2 | Details scroll 01 | medium_d64-tq64-tk128-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 4ee198976230 |
| 3 | Details scroll 02 | medium_d64-tq64-tk128-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 1a37e7c68594 |
| 4 | Details scroll 03 | medium_d64-tq64-tk128-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 5e5564b402e3 |
| 5 | Details scroll 04 | medium_d64-tq64-tk128-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 1cafe5b691d8 |
| 6 | Details scroll 05 | medium_d64-tq64-tk128-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 13ec42baca1a |
| 7 | Details scroll 06 | medium_d64-tq64-tk128-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 5bf94186019c |
| 8 | Details scroll 07 | medium_d64-tq64-tk128-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 2f81f9f25a12 |
| 9 | Details scroll 08 | medium_d64-tq64-tk128-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 62ea3ef31578 |
| 10 | Details scroll 09 | medium_d64-tq64-tk128-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.34%, achieved occupancy 10.46%, shared memory 49.15 KiB. | 1c78f2155a29 |
| 11 | Source scroll 01 | medium_d64-tq64-tk128-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 84f3427db22d |
| 12 | Source scroll 02 | medium_d64-tq64-tk128-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | f97a1c3c8252 |
| 13 | Source scroll 03 | medium_d64-tq64-tk128-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 546e5c70687e |
| 14 | Source scroll 04 | medium_d64-tq64-tk128-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 56c4facda968 |
| 15 | Source scroll 05 | medium_d64-tq64-tk128-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | dac89bfb2be8 |
| 16 | Source high-level tail | medium_d64-tq64-tk128-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | a5b44544fef8 |
| 17 | Source high-level head | medium_d64-tq64-tk128-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | c975cfb372d4 |
| 18 | SASS at selected line 91 | medium_d64-tq64-tk128-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | 3f11a8815bcd |
| 19 | Source left code horizontal | medium_d64-tq64-tk128-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | e361a417422d |
| 20 | Source left metrics horizontal | medium_d64-tq64-tk128-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | 299e6a41d75e |
| 21 | Source right code horizontal attempt | medium_d64-tq64-tk128-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 8c89eb06f183 |
| 22 | Source right metrics horizontal attempt | medium_d64-tq64-tk128-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 8c89eb06f183 |
| 23 | Source view picker | medium_d64-tq64-tk128-lat0-source-view-picker.png | Right-side view selector state. | 946135f9a3b8 |
| 24 | Source view with PTX selected | medium_d64-tq64-tk128-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | dc4ba6b1d324 |
| 25 | PTX at selected line 91 | medium_d64-tq64-tk128-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | dc4ba6b1d324 |
| 26 | Initial PTX horizontal code view | medium_d64-tq64-tk128-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 09bb7135d6e8 |
| 27 | Initial PTX horizontal metrics view | medium_d64-tq64-tk128-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | 0d967fce6456 |
| 28 | Bottom tab: Instructions & Dependencies | medium_d64-tq64-tk128-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 8b596f554434 |
| 29 | Bottom tab: Inline Functions | medium_d64-tq64-tk128-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | f42641824153 |
| 30 | Bottom tab: Source Markers | medium_d64-tq64-tk128-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | e8342d48ab30 |
| 31 | Bottom tab: Statistics | medium_d64-tq64-tk128-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 3017a7f94997 |
| 32 | Raw metrics | medium_d64-tq64-tk128-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 9854f1a56ec4 |
| 33 | PTX supplement: left code | medium_d64-tq64-tk128-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 40fecce5841a |
| 34 | PTX supplement: left metrics | medium_d64-tq64-tk128-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 4cb214749d81 |
| 35 | PTX supplement: right code | medium_d64-tq64-tk128-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | 4cb214749d81 |
| 36 | PTX supplement: right metrics | medium_d64-tq64-tk128-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | 4cb214749d81 |

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
