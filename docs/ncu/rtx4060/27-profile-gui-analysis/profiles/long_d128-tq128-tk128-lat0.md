# long_d128-tq128-tk128-lat0

**Workload:** long_d128 | **TQÃ—TK:** 128Ã—128 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report long_d128-tq128-tk128-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 23.45 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 449.22 Âµs (0.449 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 16, 8) / (128, 1, 1) | 128 CTAs; 16,384 threads. |
| Compute / memory / DRAM throughput | 17.89% / 37.43% / 12.34% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 30.35% / 37.43% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 32.54 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 94.55% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 33.84% / 14.51% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 98.30 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.33% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 88.99% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.11 / 0.11 / 1 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 9.12 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 88.99% and Eligible Warps Per Scheduler is 0.11. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 98.30 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 12.34% and bandwidth 32.54 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 62.57% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 9.1 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, but only an average of 0.11 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. |
| TheoreticalOccupancy | 62.57% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| CPIStall | 33.34% | On average, each warp of this workload spends 3.0 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory. Stall samples are reported on the consumer instruction waiting for the result, not necessarily on the producer. When this reason dominates, the kernel is memory-latency-bound, which is not necessarily memory-bandwidth-bound; prefer latency-hiding changes such as ILP, unrolling, or pipelining when bandwidth is not saturated. This stall type represents about 33.3% of the total average of 9.1 cycles between issuing two instructions. |
| WorkloadImbalance | 11.13% | One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 12.57% above the average, while the minimum instance value is 10.06% below the average. |
| WorkloadImbalance | 11.13% | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 12.57% above the average, while the minimum instance value is 10.06% below the average. |
| WorkloadImbalance | 10.03% | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 11.37% above the average, while the minimum instance value is 9.94% below the average. |

## 5. PTX/SASS evidence
- PTX export: long_d128-tq128-tk128-lat0/source-tileir-ptx.txt; 6,992 text lines, 2,825 parsed source-correlated instruction rows.
- SASS export: long_d128-tq128-tk128-lat0/source-tileir-sass.txt; 8,597 text lines, 6,866 parsed disassembly rows.
- Frequent PTX mnemonics: fma.rn.f32 (400), mul.rn.f32 (388), add.rn.f32 (272), shl.b32 (142), max.f32 (136), ex2.approx.ftz.f32 (132), fma.rm.f32 (132), sub.rn.f32 (132), mov.b32 (129), cvt.rn.f16x2.f32 (128).
- Frequent SASS mnemonics: ffma (915), mov (800), hmma.16816.f32 (512), fadd (404), fmul (388), stl (296), ldl.lu (288), ldl (247), fmnmx (136), ffma.rm (133).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | long_d128-tq128-tk128-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 449.22 Âµs, grid (1, 16, 8), compute 17.89%, DRAM 12.34%. | 492d8e3e1ddf |
| 2 | Details scroll 01 | long_d128-tq128-tk128-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 541261974cf9 |
| 3 | Details scroll 02 | long_d128-tq128-tk128-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 03bf8fd000da |
| 4 | Details scroll 03 | long_d128-tq128-tk128-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 9c79c48c63ca |
| 5 | Details scroll 04 | long_d128-tq128-tk128-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 3cfaa51e6635 |
| 6 | Details scroll 05 | long_d128-tq128-tk128-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 6ad4051ef506 |
| 7 | Details scroll 06 | long_d128-tq128-tk128-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | c9a498b552d8 |
| 8 | Details scroll 07 | long_d128-tq128-tk128-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 5870a1d44873 |
| 9 | Details scroll 08 | long_d128-tq128-tk128-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 8cfdccb2862a |
| 10 | Details scroll 09 | long_d128-tq128-tk128-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 88.99%, achieved occupancy 8.33%, shared memory 98.30 KiB. | 3ef755f616ee |
| 11 | Source scroll 01 | long_d128-tq128-tk128-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 7b3db4091507 |
| 12 | Source scroll 02 | long_d128-tq128-tk128-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | d88687f6fa24 |
| 13 | Source scroll 03 | long_d128-tq128-tk128-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 35e7225a570c |
| 14 | Source scroll 04 | long_d128-tq128-tk128-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 0e1b69df9e2f |
| 15 | Source scroll 05 | long_d128-tq128-tk128-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | dc7df53aa25c |
| 16 | Source high-level tail | long_d128-tq128-tk128-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | 09d995219f22 |
| 17 | Source high-level head | long_d128-tq128-tk128-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | ca8a087980b2 |
| 18 | SASS at selected line 91 | long_d128-tq128-tk128-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | b1b0525f1502 |
| 19 | Source left code horizontal | long_d128-tq128-tk128-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | ce7ba6e3ef1b |
| 20 | Source left metrics horizontal | long_d128-tq128-tk128-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | 964419af1ec5 |
| 21 | Source right code horizontal attempt | long_d128-tq128-tk128-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | c5eb72f1f8b4 |
| 22 | Source right metrics horizontal attempt | long_d128-tq128-tk128-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | c5eb72f1f8b4 |
| 23 | Source view picker | long_d128-tq128-tk128-lat0-source-view-picker.png | Right-side view selector state. | 7a3af9ab889a |
| 24 | Source view with PTX selected | long_d128-tq128-tk128-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 8e574a7558b8 |
| 25 | PTX at selected line 91 | long_d128-tq128-tk128-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 8e574a7558b8 |
| 26 | Initial PTX horizontal code view | long_d128-tq128-tk128-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | 3da119a1df59 |
| 27 | Initial PTX horizontal metrics view | long_d128-tq128-tk128-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | 7e3238bf4278 |
| 28 | Bottom tab: Instructions & Dependencies | long_d128-tq128-tk128-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 8ec8fead18d3 |
| 29 | Bottom tab: Inline Functions | long_d128-tq128-tk128-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | 44b2ccaf9921 |
| 30 | Bottom tab: Source Markers | long_d128-tq128-tk128-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | 30b289a74863 |
| 31 | Bottom tab: Statistics | long_d128-tq128-tk128-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | 0cb46f465d1b |
| 32 | Raw metrics | long_d128-tq128-tk128-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | 4b73f443c662 |
| 33 | PTX supplement: left code | long_d128-tq128-tk128-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 47598da722af |
| 34 | PTX supplement: left metrics | long_d128-tq128-tk128-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | 862949488ab1 |
| 35 | PTX supplement: right code | long_d128-tq128-tk128-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | fa4c75a08962 |
| 36 | PTX supplement: right metrics | long_d128-tq128-tk128-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | fa4c75a08962 |

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

**Review status:** all 36 images were opened and visually inspected individually at 1936×1048. The following notes distinguish what the GUI actually displays from interpretation; clipped symbols are marked as clipped.

### Summary and Details captures

1. **Summary** — result ID 851 / kernel; grid (1,16,8), block (128,1,1); duration 449.22 µs; 11,117,392 cycles; RTX 4060 at 2.49 GHz; compute capability 8.9; 24 SMs. Result row: estimated speedup 62.57%, runtime improvement 281.07 µs, compute throughput 17.89%, memory throughput 37.43%, 255 registers/thread. The NCU recommendation panel says theoretical occupancy 8.3% is limited by required shared memory.
2. **Details 01** — Speed of Light: compute 17.89%, memory 37.43%, L1/TEX 30.35%, L2 37.43%, DRAM 12.34%; duration 449.22 µs; elapsed cycles 11,117,392; SM active cycles 93,753,475; SM and DRAM frequencies 2.49 and 8.24 GHz. Memory: 32.54 GB/s, L1/TEX hit 8.39%, L2 hit 94.55%, compression sectors/ratio/success 0, persisting L2 size 4.72 MiB, sector promotion misses 0, Mem Busy 33.84%, Max Bandwidth 37.43%, Mem Pipes Busy 14.51%, local/shared spill requests and overhead 0. Scheduler: active/eligible/issued warps per scheduler 1.00/0.11/0.11, No Eligible 88.99%, One or More Eligible 11.01%; issue interval about 9.1 cycles.
3. **Details 02** — same memory/scheduler data; Warp State shows 9.12 cycles per issued instruction and 9.12 per executed instruction, 32.00 average active threads/warp, 31.74 average non-predicated-off threads/warp. NCU warns optional smsp__pcsamp_sample_count was not found; Long Scoreboard rule begins with estimate 33.34%.
4. **Details 03** — Long Scoreboard explanation: each warp averages about 3.0 cycles stalled on an L1TEX scoreboard dependency. The report says to locate the producer, optimize access patterns/coalescing/cache, possibly move frequently used data to shared memory, or use ILP/unrolling/pipelining when bandwidth is not saturated. It explicitly cautions that stall samples are attributed to the consumer instruction, not necessarily the producer.
5. **Details 04** — launch grid 128, 255 registers/thread, block 128, 16,384 threads, 5.33 waves/SM, Green Context 0, 24 SMs, enabled TPC IDs all.
6. **Details 05** — Launch values: CachePreferNone; static shared memory 98.30 KiB/block; dynamic shared 0; driver shared 1.02 KiB/block; shared-memory configuration 102.40 KiB; stack size 1,264; 12 TPCs. Occupancy heading starts below.
7. **Details 06** — theoretical occupancy 8.33%, theoretical active warps/SM 4, achieved occupancy 8.33%, achieved active warps/SM 4.00. CTA block limits: registers 2, shared memory 1, warps 12, SM 24. NCU theoretical occupancy rule attributes the limit to shared memory.
8. **Details 07** — occupancy remains 8.33%; GPU/Memory Distribution begins. Average SM and L1 active cycles 9,735,546.75; average L2 active cycles 8,181,133.67; DRAM active cycles 456,856; SMSP distribution shown in right column.
9. **Details 08** — distribution continuation: average SMSP active cycles 9,669,220.43; total L1 elapsed 2,638,815,504; total SMSP elapsed 10,552,016; total SM elapsed 2,631,801,504; total L2 elapsed 1,108,107,764; total DRAM elapsed 1,489,088. SM/L1 workload imbalance estimate 11.13%, max instance 12.57% above average and min 10.06% below average.
10. **Details 09** — final occupancy/distribution rows; SMSP imbalance estimate 10.03%, max 11.37% above average and min 9.94% below. NCU footer offers report customization and Copilot; these are UI controls, not profile metrics.

### Source, instruction and pane captures

11. **Source 01** — Source tab; Left View=Source, Right View=SASS; left selected file _stub.py and right selected source kernel; both navigation selectors show Instructions Executed. The visible left code begins with the generated Python protocol wrapper/imports; right begins with the initial SASS setup rows. Lower tab Instructions & Dependencies uses Scoreboard Dependencies and has empty self/input/output sections.
12. **Source 02** — left pane scrolls through ScalarProtocol declarations; right remains at initial SASS instructions. This capture advances source lines but not the SASS instruction window.
13. **Source 03** — further _stub.py protocol declarations and same SASS prefix; right category pane displays low instruction percentages for Movement, Miscellaneous, Uniform Data and Integer entries.
14. **Source 04** — further source wrapper methods; same leading SASS setup region; empty lower dependency table persists.
15. **Source 05** — later stub definitions in left pane; right SASS is still at the opening code rows. These five source scroll screenshots therefore document source vertical movement, not the full disassembly.
16. **Source high-level tail** — _stub.py lines around 72–95 show __rsub__, __rmul__, __rtruediv__, __rfloordiv__, __rmod__, __rpow__ decorated with @function, returning TileOrScalar with ellipsis bodies.
17. **Source high-level head** — earlier/later bookend around lines 63–86 shows __xor__, __radd__, __rsub__, __rmul__, __rtruediv__, __rfloordiv__ declarations. Both bookends remain generated stub/protocol code.
18. **SASS selected-line** — a selected stub/source line is correlated to the right SASS setup block. In the inspected view, line 91 is not the inner attention score/P×V loop; it corresponds to block-ID/index setup. Do not infer inner-loop mapping from the filename suffix.
19. **Horizontal left-code** — left source pane shifted horizontally so beginnings of the methods are cut; visible portions are the right ends of TileOrScalar expressions.
20. **Horizontal left-metrics** — left pane shifted to expose Instructions Executed, Attributed Stalls and Attributed Live Registers columns. These cells are blank in this correlation view.
21. **Horizontal right-code attempt** — SASS table shows the same initial code region with horizontal displacement, but its screenshot is byte-identical to the old right-metrics image. The original files do not demonstrate separate right-pane subpanel positions.
22. **Horizontal right-metrics attempt** — byte-identical to item 21; distinct right-metrics displacement is not demonstrated by this legacy capture.
23. **Source view picker** — Left View Source and Right View SASS selectors are visible, along with the toolbar and the split code/metric region. It is the SASS state before switching.
24. **PTX selected** — Right View PTX; right source selector _stub.py,cutile_attention.py. Visible PTX module header: .version 9.4, .target sm_89, .address_size 64, visible global launch metadata symbol, entry kernel symbol and long .param declarations. The entry name and parameter types are clipped horizontally.
25. **PTX selected-line** — PTX pane still shows header/parameter definitions, while the left source remains generated stub code. This is not an instruction body/attention-loop view.
26. **Initial PTX horizontal code** — PTX/source split remains at the generated module signature; the long parameter string is cut off at the viewport.
27. **Initial PTX horizontal metrics** — left attribution columns include Instructions Executed, Attributed Stalls and Attributed Live Registers; right source shows the generated PTX parameter block.
28. **Instructions & Dependencies** — Scoreboard Dependencies dropdown selected; lower tables Self Instruction Mix, Input Scoreboard Dependencies and Output Scoreboard Dependencies. Visible headers Instruction Category, Scoreboard Stalls, All Stalls, Instructions, Location, Scoreboards and Attributed Stalls; no rows are filled.
29. **Inline Functions** — lower panel message reads “Select a source line in an active inline function to show additional information.” This does not establish that compiled code has no inline functions.
30. **Source Markers** — filter All; columns File, Line/Address, Marker and Source; no marker rows.
31. **Statistics** — Aggregate, Maximum, Minimum and Average rows, columns Live Registers, Instruction Mix, Instructions Executed, Attributed Stalls and Attributed Live Registers; no metric values populated.
32. **Raw** — ID 0; time 2026-Oct-31 19:25:58; API Call ID 851; estimated speedup 62.57%; runtime improvement 281.07 µs; 9 issues; function kernel; process [7008] python.exe; thread 21564; RTX 4060; grid (1,16,8), block (128,1,1), 3 grid dimensions; 11,117,392 cycles; GPU duration 449.22 µs; compute 17.89%, memory 37.43%; spill requests 0. Mangled/demangled names and addresses are truncated in the GUI cells.
33. **PTX supplement — left code** — opening _stub.py source: NVIDIA copyright, Apache-2.0 SPDX, future annotations, functools/inspect/textwrap/dataclass/typing, CUDA Tile memory-model/execution/datatype/numeric-semantics imports, Types and ScalarProtocol.
34. **PTX supplement — left metrics** — same initial source with Instructions Executed, Attributed Stalls and Attributed Live Registers columns visible.
35. **PTX supplement — right code** — PTX mode horizontally shifted into the long generated kernel parameter signature; the left edge of the visible parameter list is different from the original PTX view.
36. **PTX supplement — right metrics** — distinct PTX-mode parameter pane crop; long generated text continues beyond the right edge. The pane-local corrected proof is maintained separately from the old capture pair.

### Visible SASS instruction transcription

The first 23 visible SASS instructions/rows in the Source screenshots are:

1. MOV R1, c[0x0][0x28]
2. S2R R6, SR_TID.X
3. MOV R7, c[0x0][0x28]
4. S2UR UR4, SR_CTAID.X
5. ULDC.64 UR6, c[0x0][0x118]
6. S2R R8, SR_CTAID.Z
7. IADD3 R0, P2, R7, c[0x0][0x174], …
8. CS2R R28, SRZ
9. IADD3 R7, P0, R7, c[0x0][0x170], …
10. S2R R69, SR_CTAID.Y
11. LOP3.LUT P1, RZ, R0, 0xffffffff, …
12. CS2R R30, SRZ
13. IADD3 R1, R1, -0x4e8, RZ
14. CS2R R20, SRZ
15. CS2R R22, SRZ
16. CS2R R18, SRZ
17. CS2R R12, SRZ
18. CS2R R14, SRZ
19. CS2R R32, SRZ
20. CS2R R34, SRZ
21. CS2R R36, SRZ
22. CS2R R38, SRZ
23. CS2R R24, SRZ

The instruction category chart lists Movement, Miscellaneous, Integer and Uniform Data with displayed percentages below 0.01%. The long operands are clipped; no hidden operand text is inferred.

### Per-configuration interpretation

This TQ128/TK128 profile has a 98.30 KiB static shared-memory footprint per CTA. The shared-memory block limit is one CTA/SM, yielding 8.33% theoretical and achieved occupancy (4 active warps/SM). Scheduler readiness is sparse (0.11 eligible warps/scheduler; No Eligible 88.99%). NCU Long Scoreboard attributes about 3.0 cycles/warp to an L1TEX dependency, but its samples identify the waiting consumer and not necessarily the producer. DRAM throughput is 12.34%, so the screenshot does not indicate saturated DRAM bandwidth. Spill requests are zero. Source attribution is _stub.py, not the high-level attention kernel.
