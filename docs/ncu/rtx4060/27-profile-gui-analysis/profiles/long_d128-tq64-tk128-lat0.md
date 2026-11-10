# long_d128-tq64-tk128-lat0

**Workload:** long_d128 | **TQÃ—TK:** 64Ã—128 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report long_d128-tq64-tk128-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 23.16 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 297.47 Âµs (0.297 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 16, 16) / (128, 1, 1) | 256 CTAs; 32,768 threads. |
| Compute / memory / DRAM throughput | 26.57% / 18.38% / 16.11% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 19.62% / 18.37% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 42.50 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 84.99% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 18.38% / 9.94% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 81.92 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.30% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 3.98 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 85.69% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.14 / 0.14 / 1 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 6.96 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 85.69% and Eligible Warps Per Scheduler is 0.14. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 81.92 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 16.11% and bandwidth 42.50 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 73.43% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 7.0 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, which already limits the scheduler to less than a warp per instruction. |
| TheoreticalOccupancy | 73.43% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| CPIStall | 42.01% | On average, each warp of this workload spends 2.9 cycles being stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. When this reason dominates, the kernel is usually compute-pipeline-bound; otherwise, increasing active warps can help hide latency. This stall type represents about 42.0% of the total average of 7.0 cycles between issuing two instructions. |
| WorkloadImbalance | 5.48% | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 5.88% above the average, while the minimum instance value is 6.08% below the average. |
| WorkloadImbalance | 5.31% | One or more SMs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 5.67% above the average, while the minimum instance value is 6.11% below the average. |
| WorkloadImbalance | 5.31% | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Additionally, other L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 5.67% above the average, while the minimum instance value is 6.11% below the average. |

## 5. PTX/SASS evidence
- PTX export: long_d128-tq64-tk128-lat0/source-tileir-ptx.txt; 4,519 text lines, 1,698 parsed source-correlated instruction rows.
- SASS export: long_d128-tq64-tk128-lat0/source-tileir-sass.txt; 4,758 text lines, 3,416 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (200), mul.rn.f32 (194), add.rn.f32 (136), mov.b32 (97), shl.b32 (87), max.f32 (68), ex2.approx.ftz.f32 (66), fma.rm.f32 (66), sub.rn.f32 (66), add.s32 (64).
- Frequent SASS mnemonics: ffma (461), hmma.16816.f32 (256), mov (235), fadd (202), fmul (194), prmt (128), lop3.lut (113), iadd3 (110), cs2r (96), imad.mov.u32 (80).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | long_d128-tq64-tk128-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 297.47 Âµs, grid (1, 16, 16), compute 26.57%, DRAM 16.11%. | 2c4ddd83fcfb |
| 2 | Details scroll 01 | long_d128-tq64-tk128-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | 422d438c616c |
| 3 | Details scroll 02 | long_d128-tq64-tk128-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | e4d64c7acf38 |
| 4 | Details scroll 03 | long_d128-tq64-tk128-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | b2e7effd3933 |
| 5 | Details scroll 04 | long_d128-tq64-tk128-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | 3cfc584f1ae9 |
| 6 | Details scroll 05 | long_d128-tq64-tk128-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | 50f45c90e496 |
| 7 | Details scroll 06 | long_d128-tq64-tk128-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | 4f6f44406221 |
| 8 | Details scroll 07 | long_d128-tq64-tk128-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | 4fe5595bc68f |
| 9 | Details scroll 08 | long_d128-tq64-tk128-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | aa5c6c0068e9 |
| 10 | Details scroll 09 | long_d128-tq64-tk128-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.69%, achieved occupancy 8.30%, shared memory 81.92 KiB. | 21123c61b047 |
| 11 | Source scroll 01 | long_d128-tq64-tk128-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 0c894f0a1c59 |
| 12 | Source scroll 02 | long_d128-tq64-tk128-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 4db824cea5c4 |
| 13 | Source scroll 03 | long_d128-tq64-tk128-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 123701bdeab4 |
| 14 | Source scroll 04 | long_d128-tq64-tk128-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 458ddd3fdd1a |
| 15 | Source scroll 05 | long_d128-tq64-tk128-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | a3d313c26fa8 |
| 16 | Source high-level tail | long_d128-tq64-tk128-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | bdc38cd152f6 |
| 17 | Source high-level head | long_d128-tq64-tk128-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | bf247f520990 |
| 18 | SASS at selected line 91 | long_d128-tq64-tk128-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | 5bdcfa70d55f |
| 19 | Source left code horizontal | long_d128-tq64-tk128-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | 0df911dbcc27 |
| 20 | Source left metrics horizontal | long_d128-tq64-tk128-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | 9ecc51e38467 |
| 21 | Source right code horizontal attempt | long_d128-tq64-tk128-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | d3dcc1c8c942 |
| 22 | Source right metrics horizontal attempt | long_d128-tq64-tk128-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | d3dcc1c8c942 |
| 23 | Source view picker | long_d128-tq64-tk128-lat0-source-view-picker.png | Right-side view selector state. | 91037d377cfb |
| 24 | Source view with PTX selected | long_d128-tq64-tk128-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 395b8673909b |
| 25 | PTX at selected line 91 | long_d128-tq64-tk128-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 395b8673909b |
| 26 | Initial PTX horizontal code view | long_d128-tq64-tk128-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 00d0e1f36c27 |
| 27 | Initial PTX horizontal metrics view | long_d128-tq64-tk128-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | e6c5fa92bc54 |
| 28 | Bottom tab: Instructions & Dependencies | long_d128-tq64-tk128-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | f882e4708184 |
| 29 | Bottom tab: Inline Functions | long_d128-tq64-tk128-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 5341458e5cf7 |
| 30 | Bottom tab: Source Markers | long_d128-tq64-tk128-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | d6cdd274a7d9 |
| 31 | Bottom tab: Statistics | long_d128-tq64-tk128-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 320d830052dc |
| 32 | Raw metrics | long_d128-tq64-tk128-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | d81b0c9fba2a |
| 33 | PTX supplement: left code | long_d128-tq64-tk128-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 6264a76f0800 |
| 34 | PTX supplement: left metrics | long_d128-tq64-tk128-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | b452f5f675f9 |
| 35 | PTX supplement: right code | long_d128-tq64-tk128-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | 8108774675e5 |
| 36 | PTX supplement: right metrics | long_d128-tq64-tk128-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | 8108774675e5 |

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
