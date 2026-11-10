# long_d128-tq32-tk64-lat0

**Workload:** long_d128 | **TQÃ—TK:** 32Ã—64 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report long_d128-tq32-tk64-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 23.20 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 358.40 Âµs (0.358 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 16, 32) / (128, 1, 1) | 512 CTAs; 65,536 threads. |
| Compute / memory / DRAM throughput | 20.78% / 27.40% / 13.39% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 22.28% / 27.40% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 35.33 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 91.63% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 27.40% / 16.30% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 78.08 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.33% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 81.04% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.19 / 0.19 / 1 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 5.27 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 81.04% and Eligible Warps Per Scheduler is 0.19. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 78.08 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 13.39% and bandwidth 35.33 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 72.60% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 5.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, which already limits the scheduler to less than a warp per instruction. |
| TheoreticalOccupancy | 72.60% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |

## 5. PTX/SASS evidence
- PTX export: long_d128-tq32-tk64-lat0/source-tileir-ptx.txt; 6,149 text lines, 2,556 parsed source-correlated instruction rows.
- SASS export: long_d128-tq32-tk64-lat0/source-tileir-sass.txt; 5,899 text lines, 4,061 parsed disassembly rows.
- Frequent PTX mnemonics: mov.b32 (310), mul.rn.f32 (272), fma.rn.f32 (256), add.rn.f32 (176), add.s32 (135), shl.b32 (104), max.f32 (96), selp.b32 (88), xor.b32 (82), ex2.approx.ftz.f32 (80).
- Frequent SASS mnemonics: ffma (343), prmt (304), fmul (272), fadd (256), hmma.16816.f32 (256), imad.mov.u32 (192), lop3.lut (190), iadd3 (140), fmnmx (96), imad.shl.u32 (95).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | long_d128-tq32-tk64-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 358.40 Âµs, grid (1, 16, 32), compute 20.78%, DRAM 13.39%. | e5e4841bc417 |
| 2 | Details scroll 01 | long_d128-tq32-tk64-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | b5d03b350114 |
| 3 | Details scroll 02 | long_d128-tq32-tk64-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 96e94cfba426 |
| 4 | Details scroll 03 | long_d128-tq32-tk64-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 865d1dbe42d3 |
| 5 | Details scroll 04 | long_d128-tq32-tk64-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | a38e0836545a |
| 6 | Details scroll 05 | long_d128-tq32-tk64-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 2f8e338a149e |
| 7 | Details scroll 06 | long_d128-tq32-tk64-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | ad58ec8c5a64 |
| 8 | Details scroll 07 | long_d128-tq32-tk64-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | e47ac9a61471 |
| 9 | Details scroll 08 | long_d128-tq32-tk64-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 0f962a805bee |
| 10 | Details scroll 09 | long_d128-tq32-tk64-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 81.04%, achieved occupancy 8.33%, shared memory 78.08 KiB. | 0f962a805bee |
| 11 | Source scroll 01 | long_d128-tq32-tk64-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 88171305d20f |
| 12 | Source scroll 02 | long_d128-tq32-tk64-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 066f4e1ef11b |
| 13 | Source scroll 03 | long_d128-tq32-tk64-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | e9a344364c63 |
| 14 | Source scroll 04 | long_d128-tq32-tk64-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | c94891f4bfda |
| 15 | Source scroll 05 | long_d128-tq32-tk64-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | d845eedfe879 |
| 16 | Source high-level tail | long_d128-tq32-tk64-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | 181a8c684061 |
| 17 | Source high-level head | long_d128-tq32-tk64-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | ebb9c4ec28ee |
| 18 | SASS at selected line 91 | long_d128-tq32-tk64-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | 05db35923139 |
| 19 | Source left code horizontal | long_d128-tq32-tk64-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | 3597aeb69cc3 |
| 20 | Source left metrics horizontal | long_d128-tq32-tk64-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | fae165bacada |
| 21 | Source right code horizontal attempt | long_d128-tq32-tk64-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 214ebca2f7fc |
| 22 | Source right metrics horizontal attempt | long_d128-tq32-tk64-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 214ebca2f7fc |
| 23 | Source view picker | long_d128-tq32-tk64-lat0-source-view-picker.png | Right-side view selector state. | c8f15f558312 |
| 24 | Source view with PTX selected | long_d128-tq32-tk64-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 80e7516d7007 |
| 25 | PTX at selected line 91 | long_d128-tq32-tk64-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 80e7516d7007 |
| 26 | Initial PTX horizontal code view | long_d128-tq32-tk64-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 3aac6bd4fc2e |
| 27 | Initial PTX horizontal metrics view | long_d128-tq32-tk64-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | c59e6c310abe |
| 28 | Bottom tab: Instructions & Dependencies | long_d128-tq32-tk64-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | a645605d925b |
| 29 | Bottom tab: Inline Functions | long_d128-tq32-tk64-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | cdd5827753b1 |
| 30 | Bottom tab: Source Markers | long_d128-tq32-tk64-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | 509877c56535 |
| 31 | Bottom tab: Statistics | long_d128-tq32-tk64-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | de04dce6c0be |
| 32 | Raw metrics | long_d128-tq32-tk64-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 999cdaa7c7b8 |
| 33 | PTX supplement: left code | long_d128-tq32-tk64-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 4e71eeca97bf |
| 34 | PTX supplement: left metrics | long_d128-tq32-tk64-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 7855695cff29 |
| 35 | PTX supplement: right code | long_d128-tq32-tk64-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | 6995bf2500e1 |
| 36 | PTX supplement: right metrics | long_d128-tq32-tk64-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | 6995bf2500e1 |

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

## 8. Manual image-by-image transcription (native-resolution review)

**Review status:** all 36 screenshots (32 primary plus four PTX supplements) were opened and visually inspected individually at 1936×1048.

### Summary and Details captures

1. **Summary** — result 851 / kernel; grid (1,16,32), block (128,1,1); NCU duration 358.40 µs; RTX 4060 at 2.49 GHz; process [15220] python.exe; compute capability 8.9, 24 SMs. Result row: estimated speedup 72.60%, runtime improvement 260.19 µs, compute throughput 20.78%, memory throughput 27.40%, 255 registers/thread.
2. **Details 01** — compute 20.78%, memory 27.40%, L1/TEX 22.28%, L2 27.40%, DRAM 13.39%; duration 358.40 µs; SM active cycles 8,58,105.29; frequencies 2.49/8.24 GHz. Memory throughput 35.33 GB/s; L1/TEX hit 0%; L2 hit 91.63%; compression counters 0; persisting size 4.72 MiB; Mem Busy 27.40%; Max Bandwidth 26.64%; Mem Pipes Busy 16.30%; local/shared spill requests and overhead 0. Scheduler active 1.00, eligible/issued 0.19; No Eligible 81.04%, One or More Eligible 18.96%; Issue Slot Utilization estimate 72.60%.
3. **Details 02** — warp cycles per issued and executed instruction 5.27; average active threads/warp 31.49; average non-predicated-off threads/warp 30.77. Warning: optional smsp__pcsamp_sample_count metric could not be found. No scoreboard/math-pipe stall rule is displayed.
4. **Details 03** — repeats Memory Workload and Scheduler values; warp-state values and missing optional PC-sampling warning remain visible.
5. **Details 04** — launch grid 512, registers/thread 255, block 128, threads 65,536, waves/SM 21.33, Green Context 0, 24 SMs, all TPC IDs. Static shared memory 78.08 KiB/block; dynamic shared 0; driver shared 1.02 KiB/block; configured shared memory 102.40 KiB; stack 1,024; 12 TPCs.
6. **Details 05** — launch summary is repeated with occupancy section starting below. Theoretical occupancy 8.33%, theoretical active warps/SM 4, achieved occupancy 8.33%, achieved active warps/SM 4.00; register block limit 2, shared-memory block limit 1, warps block limit 12 and SM limit 24.
7. **Details 06** — issue-slot rule explanation says each scheduler issues an instruction every 5.3 cycles and has 1.00 active warp/scheduler; theoretical occupancy remains 8.3% because of shared memory. Distribution begins: average SM/L1 active cycles 8,58,105.29; average L2 5,92,967.83; DRAM 3,95,680; SMSP 8,58,565.61.
8. **Details 07** — distribution totals: total SM elapsed 2,14,42,344; total L1 elapsed same; total L2 86,27,124; total SMSP 8,57,69,376; total DRAM 1,18,16,960. The view includes the report-customization/Copilot footer.
9. **Details 08** — lower warp/launch/occupancy summary and full distribution; no additional stall rule is introduced.
10. **Details 09** — same tail as Details 08: occupancy 8.33%, active warps/SM 4, shared-memory CTA limit 1, and same distribution totals. This is scroll overlap, not a second measurement.

### Source, instruction and pane captures

11. **Source 01** — Left View Source (file _stub.py), Right View SASS (source kernel), Navigate By Instructions Executed. Left code around lines 72–94; right SASS is an integer/uniform-data-heavy setup sequence. Bottom Instructions & Dependencies / Scoreboard Dependencies pane has empty self/input/output tables.
12. **Source 02** — source around lines 81–103; visible protocol methods include __rfloordiv__, __rmod__, __rpow__, __rand__, __ror__, __rxor__. Right SASS remains at the same initial instruction prefix.
13. **Source 03** — source around lines 90–113; methods include __rpow__, __rand__, __ror__, __rxor__, __ge__, __gt__.
14. **Source 04** — source around lines 99–122; methods include __ror__, __rxor__, __ge__, __gt__, __le__, __lt__.
15. **Source 05** — source around lines 108–131; methods include __ge__, __gt__, __le__, __lt__, __eq__, __ne__.
16. **Source high-level head** — _stub.py lines around 99–122 repeat __ror__, __rxor__, __ge__, __gt__, __le__, __lt__. The left source is a Tile scalar protocol wrapper, not the project attention body.
17. **Source high-level tail** — lines around 108–131 include __ge__, __gt__, __le__, __lt__, __eq__, __ne__.
18. **SASS selected-line** — highlight is on approximately source line 114 in the visible _stub.py listing. The right pane remains the generated kernel integer CTA/linear-index setup; the filename suffix line91 is not proof that line 91 is selected.
19. **Horizontal left-code** — left source is shifted so only tails of TileOrScalar method signatures remain; beginnings are clipped.
20. **Horizontal left-metrics** — columns Instructions Executed, Attributed Stalls and Attributed Live Registers are exposed; cells are blank.
21. **Horizontal right-code** — right SASS pane shows the integer/uniform-data setup sequence; the old capture shares its hash with item 22.
22. **Horizontal right-metrics** — same legacy image as item 21; original capture does not prove independent right-pane movement.
23. **Source view picker** — shows Right View SASS and code/category pane divisions; left source is horizontally clipped at TileOrScalar string tails.
24. **PTX view** — Right View PTX; source selector _stub.py,cutile_attention.py. Header shows .version 9.4, .target sm_89, .address_size 64, global launch metadata symbol, entry symbol and long parameter declarations.
25. **PTX selected-line** — left _stub.py around lines 108–131; right PTX remains at the module header/parameter region; no attention-loop body is visible.
26. **Initial PTX horizontal code** — left source head (copyright, Apache license, Python/CUDA Tile imports) with long generated kernel parameter declaration on the right.
27. **Initial PTX horizontal metrics** — attribution metric headings on left; right PTX pane shows generated parameter list.
28. **Instructions & Dependencies** — right PTX source pane is scrolled to Tile IR around lines 7197–7220. Visible operations include @reduce_lhs, @reduce_rhs, addf, yield, reshape_128, fma, ftof f32→f16, broadcast, mulf, mmaf, continue, divf, reshape, constant<i32 0>, make_partition_view, store_view_tko, return and .source_end. Some long type operands are clipped.
29. **Inline Functions** — same Tile IR tail remains visible; lower pane says “Select a source line in an active inline function to show additional information.”
30. **Source Markers** — same Tile IR tail; All filter selected; columns File, Line/Address, Marker, Source; no rows.
31. **Statistics** — same Tile IR tail; rows Aggregate, Maximum, Minimum, Average; columns Live Registers, Instruction Mix, Instructions Executed, Attributed Stalls, Attributed Live Registers; no values.
32. **Raw** — ID 0; 2026-Oct-31 19:22:27; API Call 851; Estimated Speedup 72.60%; Runtime Improvement 260.19 µs; 4 issues; function kernel; process [15220] python.exe; thread 9928; RTX 4060; grid (1,16,32), block (128,1,1), 3 dimensions; GPU time 358.40 µs; compute 20.78%, memory 27.40%, spill requests 0. Mangled/demangled name cells and some addresses are clipped.
33. **PTX supplement — left code** — _stub.py opening lines, Apache-2.0 SPDX, future annotations, Python imports, CUDA Tile imports, Types and ScalarProtocol.
34. **PTX supplement — left metrics** — same source opening with Instructions Executed, Attributed Stalls and Attributed Live Registers columns.
35. **PTX supplement — right code** — PTX parameter list shifted to a different horizontal position; prefix/suffix clipped.
36. **PTX supplement — right metrics** — separately shifted parameter-list crop, with a different horizontal offset; generated name shows the TQ32 signature fragment.

### Visible SASS instruction transcription

The 23 displayed instruction rows in the initial SASS pane (some operands are clipped by the GUI pane boundary) are:

1. IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]
2. S2R R39, SR_TID.X
3. S2UR UR14, SR_CTAID.Z
4. ULDC.64 UR6, c[0x0][0x170]
5. CS2R R12, SRZ
6. UIADD3 UR5, UP0, UR6, 0x1f, URZ
7. CS2R R14, SRZ
8. UIADD3 UR4, UP1, UR7, 0x7f, URZ
9. CS2R R16, SRZ
10. UIADD3.X UR6, UR6, URZ, URZ, UP0, …
11. S2UR UR15, SR_CTAID.Y
12. UIADD3.X UR7, UR7, URZ, URZ, UP1, …
13. CS2R R18, SRZ
14. USHF.R.U64 UR4, UR5, 0x5, UR6
15. USHF.R.U32.HI UR6, URZ, 0x5, UR6
16. ULOP3.LUT UP1, URZ, UR7, 0xffffffff, …
17. S2UR UR17, SR_CTAID.X
18. ULDC.64 UR12, c[0x0][0x118]
19. ULOP3.LUT UP1, URZ, UR7, 0x1, URZ, …
20. ULDC.64 UR12, c[0x0][0x118]
21. SHF.R.U32.HI R6, RZ, R0, 0x5, R39
22. IMAD.SHL.U32 R38, R39, 0x80, RZ
23. SHF.L.U32 R39, R9, 0x1, RZ

The rightmost category panel classifies visible entries mainly as Integer, Miscellaneous and Uniform Data, with displayed percentages around 0.01% or less. Ellipses explicitly mark operands cut off by the pane; they are not guessed.

### Per-configuration interpretation

The smaller TQ32 tile gives a 512-CTA grid and 21.33 waves/SM, but 78.08 KiB static shared memory still limits theoretical/achieved occupancy to 8.33% / 4 active warps per SM. Eligible warps rise to 0.19 and No Eligible falls to 81.04%, while DRAM throughput remains 13.39%. The NCU 72.60% Issue Slot / Theoretical Occupancy estimate is not a measured improvement. Missing PC-sampling data is a collection limitation, not evidence of no stalls. The lower PTX source exposes the TQ32-specific Tile IR tail with mmaf, conversions, broadcasts and partition-view/store operations, but does not directly map to the high-level Python kernel.
