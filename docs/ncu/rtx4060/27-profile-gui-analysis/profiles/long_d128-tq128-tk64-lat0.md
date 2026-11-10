# long_d128-tq128-tk64-lat0

**Workload:** long_d128 | **TQÃ—TK:** 128Ã—64 | **load latency:** 0 | **status:** PASS

## 1. Provenance
- Native report long_d128-tq128-tk64-lat0.ncu-repz is in [workflow 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).
- 32 genuine GUI screenshots are in [primary capture run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 4 PTX-mode supplementary captures are in [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876).
- Hardware: NVIDIA GeForce RTX 4060, compute capability 8.9, 24 SMs, driver 616.92, 8,188 MiB reported; Nsight Compute 2026.3.1.
- Profile commit: 708a7c7838b99539298e0afd0ed330082796e60e; capture session elapsed 23.21 s (not kernel latency).
- GUI capture opened the existing .ncu-repz sequentially; it did not replay kernels or redo profiling. The native profile files were not overwritten.

## 2. Summary and launch geometry
| Item | Value | Interpretation |
|---|---:|---|
| NCU duration | 343.78 Âµs (0.344 ms) | Original metric unit: us; instrumented value, not normal CUDA-event latency. |
| Grid / block | (1, 16, 8) / (128, 1, 1) | 128 CTAs; 16,384 threads. |
| Compute / memory / DRAM throughput | 22.98% / 27.52% / 13.98% | All normalized throughput metrics from this report. |
| L1/TEX / L2 throughput | 21.98% / 27.52% | Cache-pipeline throughput indicators. |
| Memory bandwidth | 36.88 GB/s | Memory Workload Analysis throughput. |
| L2 hit rate | 91.21% | Interpret with traffic and working set. |
| Mem Busy / Mem Pipes Busy | 27.35% / 13.03% | Neither is evidence of saturated DRAM. |
| Registers / thread | 255 | Per-thread register demand. |
| Static shared memory / CTA | 98.30 KiB | Important CTA residency constraint. |
| Theoretical / achieved occupancy | 8.33% / 8.35% | Achieved is near this launch/resource cap. |
| Achieved active warps / SM | 4.01 | Average active warps per SM. |
| CTA limit from shared / registers / warps | 1 / 2 / 12 | Minimum limit controls active CTA count. |
| No Eligible cycles | 85.22% | No ready warp on these scheduler cycles; not whole-GPU idle time. |
| Eligible / issued / active warps per scheduler | 0.15 / 0.15 / 1 | Sparse issue-ready pool. |
| Warp cycles per issued instruction | 6.77 | Aggregate issue spacing, not a single dependency latency. |
| Local/shared spill requests | 0 / 0 | No spill requests reported. |

## 3. Interpretation for this configuration
- Scheduler: No Eligible is 85.22% and Eligible Warps Per Scheduler is 0.15. This indicates frequent absence of an immediately ready warp. It does not uniquely identify long scoreboard, math-pipe dependency, barrier wait or insufficient independent CTAs.
- Residency: 98.30 KiB static shared memory per CTA; the shared-memory block limit is 1 CTA(s)/SM. Where this value is 1, extra grid CTAs do not create a second concurrent CTA on each SM.
- Memory: DRAM throughput 13.98% and bandwidth 36.88 GB/s do not show DRAM saturation. Memory-latency dependencies remain possible.
- Spilling: both local and shared spill requests are zero; spilling is not supported as the cause by this report.
- Use this profile as diagnostic evidence for this precise workload/tile on SM89. Do not change production defaults without correctness and repeated unprofiled latency tests.

## 4. NCU rule-engine messages
Estimated Speedup percentages are NCU diagnostic/rule estimates, not measured runtime improvements.

| Rule | Estimated speedup | Explanation |
|---|---:|---|
| IssueSlotUtilization | 72.48% | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 6.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, but only an average of 0.15 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too. |
| TheoreticalOccupancy | 72.48% | The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of shared memory. |
| CPIStall | 33.70% | On average, each warp of this workload spends 2.3 cycles being stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. When this reason dominates, the kernel is usually compute-pipeline-bound; otherwise, increasing active warps can help hide latency. This stall type represents about 33.7% of the total average of 6.8 cycles between issuing two instructions. |
| WorkloadImbalance | 10.12% | One or more SMs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 11.49% above the average, while the minimum instance value is 11.04% below the average. |
| WorkloadImbalance | 10.12% | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Additionally, other L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 11.49% above the average, while the minimum instance value is 11.04% below the average. |
| WorkloadImbalance | 9.96% | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. Additionally, other SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 11.35% above the average, while the minimum instance value is 11.11% below the average. |

## 5. PTX/SASS evidence
- PTX export: long_d128-tq128-tk64-lat0/source-tileir-ptx.txt; 4,506 text lines, 1,846 parsed source-correlated instruction rows.
- SASS export: long_d128-tq128-tk64-lat0/source-tileir-sass.txt; 6,277 text lines, 4,896 parsed disassembly rows.
- Frequent PTX mnemonics: mul.rn.f32 (260), fma.rn.f32 (208), add.rn.f32 (144), mov.b32 (129), div.rn.f32 (128), cvt.rn.f16x2.f32 (96), shl.b32 (79), max.f32 (72), ex2.approx.ftz.f32 (68), fma.rm.f32 (68).
- Frequent SASS mnemonics: ffma (787), mov (643), fmul (260), hmma.16816.f32 (256), fadd (212), ldl (159), imad.mov.u32 (157), bssy (130), bsync (130), mufu.rcp (129).
- Across the 27 exports, PTX contains bar.sync, global/shared load/store, FMA and exponent-related operations; SASS contains HMMA tensor-core instructions, FFMA, MUFU, memory operations and BAR.SYNC. The PTX text dump does not show a literal mma token; that does not imply absence of tensor-core operations, which are visible as HMMA in SASS.
- The GUI's left source often resolves to generated CuTile stub code rather than a direct listing of the high-level kernel. In the inspected D=128 latency-10 export, source line 91 is block-ID setup, not the attention inner loop. Blank dependency/inline/source-marker panels mean no populated attribution was visible, not that those relationships are absent.

## 6. Screenshot-by-screenshot review (36 captures)
All PNG files are genuine captures from the actual Nsight Compute GUI and are 1936Ã—1048. Extract the two screenshot artifacts and open the exact files listed below.

| # | View | PNG filename | Per-image interpretation | SHA-256 prefix |
|---:|---|---|---|---|
| 1 | Summary | long_d128-tq128-tk64-lat0-summary.png | Kernel result row, duration in the native unit, launch dimensions, GPU metadata and rule-engine messages. Estimated-speedup values are diagnostic estimates, not measured gains. This profile: 343.78 Âµs, grid (1, 16, 8), compute 22.98%, DRAM 13.98%. | 9de507ab5298 |
| 2 | Details scroll 01 | long_d128-tq128-tk64-lat0-details-01.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | c15ba86ce712 |
| 3 | Details scroll 02 | long_d128-tq128-tk64-lat0-details-02.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 1898d4405c74 |
| 4 | Details scroll 03 | long_d128-tq128-tk64-lat0-details-03.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 3554181d3a63 |
| 5 | Details scroll 04 | long_d128-tq128-tk64-lat0-details-04.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 57e522bef75a |
| 6 | Details scroll 05 | long_d128-tq128-tk64-lat0-details-05.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | eefa130a0a8e |
| 7 | Details scroll 06 | long_d128-tq128-tk64-lat0-details-06.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 24b7c3628c47 |
| 8 | Details scroll 07 | long_d128-tq128-tk64-lat0-details-07.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 71db80fe52a0 |
| 9 | Details scroll 08 | long_d128-tq128-tk64-lat0-details-08.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | 3d8c7bb04803 |
| 10 | Details scroll 09 | long_d128-tq128-tk64-lat0-details-09.png | One of nine ordered, overlapping vertical scroll positions. Together they span Speed of Light, Memory Workload, Scheduler, Warp State, Launch, Occupancy and GPU/Memory Workload Distribution; CSV supplies exact values. Key values in this report: No Eligible 85.22%, achieved occupancy 8.35%, shared memory 98.30 KiB. | a8955ee410dd |
| 11 | Source scroll 01 | long_d128-tq128-tk64-lat0-source-01.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 474b26906b7e |
| 12 | Source scroll 02 | long_d128-tq128-tk64-lat0-source-02.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | c4ca3103e12c |
| 13 | Source scroll 03 | long_d128-tq128-tk64-lat0-source-03.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 60dad36e4525 |
| 14 | Source scroll 04 | long_d128-tq128-tk64-lat0-source-04.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 7ba105d72bd8 |
| 15 | Source scroll 05 | long_d128-tq128-tk64-lat0-source-05.png | Sequential vertical source/disassembly correlation. Left source resolves to generated CuTile stub content in these reports; right pane is SASS correlation, not a direct source listing of the high-level kernel module. | 9e7ff8f6f2c6 |
| 16 | Source high-level tail | long_d128-tq128-tk64-lat0-source-high-level-tail.png | Bookend view of the Source tab's higher-level source attribution. | c8015c3b4149 |
| 17 | Source high-level head | long_d128-tq128-tk64-lat0-source-high-level-head.png | Bookend view of the Source tab's higher-level source attribution. | a77c880ee1b6 |
| 18 | SASS at selected line 91 | long_d128-tq128-tk64-lat0-source-line91-sass.png | SASS correlated with selected generated source line. In the inspected D=128 profile this is tile block-ID setup, not the attention score/PV loop. | bd26792b576b |
| 19 | Source left code horizontal | long_d128-tq128-tk64-lat0-source-horizontal-left-code.png | Left code pane after a horizontal action; differs from the left-metrics capture. | 6d0a01d11820 |
| 20 | Source left metrics horizontal | long_d128-tq128-tk64-lat0-source-horizontal-left-metrics.png | Left metrics pane after a horizontal action; exposes additional instruction/stall/register columns. The visible left-side metrics columns are distinct from the left-code view. | f87123ce06a8 |
| 21 | Source right code horizontal attempt | long_d128-tq128-tk64-lat0-source-horizontal-right-code.png | Right SASS/code pane after the requested horizontal action. The SHA-256 comparison found no visible change from its paired right-side image. | 1c6f6bebd082 |
| 22 | Source right metrics horizontal attempt | long_d128-tq128-tk64-lat0-source-horizontal-right-metrics.png | SHA-256 audit finds this byte-identical to right-code for all 27 reports; no distinct horizontal displacement is demonstrated. The SHA-256 comparison found no visible change from its paired right-side image. | 1c6f6bebd082 |
| 23 | Source view picker | long_d128-tq128-tk64-lat0-source-view-picker.png | Right-side view selector state. | dffdad1b816e |
| 24 | Source view with PTX selected | long_d128-tq128-tk64-lat0-source-ptx.png | PTX is selected in the right Source view. The exports identify PTX 9.4 targeting sm_89; this capture shows the entry/parameter region. | 390da68f57ed |
| 25 | PTX at selected line 91 | long_d128-tq128-tk64-lat0-source-line91-ptx.png | PTX correlated to the selected generated source line; inspected example shows CTA/block-ID derivation. | 390da68f57ed |
| 26 | Initial PTX horizontal code view | long_d128-tq128-tk64-lat0-source-ptx-horizontal-ptx-code.png | Initial horizontal capture of the PTX-mode code-oriented pane. | b007838b78b4 |
| 27 | Initial PTX horizontal metrics view | long_d128-tq128-tk64-lat0-source-ptx-horizontal-ptx-metrics.png | Initial horizontal capture of the PTX-mode metrics-oriented pane. | aa101aa34be9 |
| 28 | Bottom tab: Instructions & Dependencies | long_d128-tq128-tk64-lat0-source-bottom-instructions-dependencies.png | Lower Source pane with the dependency selector and self/input/output tables. Blank tables mean no populated attribution in this view, not zero dependencies. | 269495ecbb29 |
| 29 | Bottom tab: Inline Functions | long_d128-tq128-tk64-lat0-source-bottom-inline-functions.png | Lower Source pane selected to Inline Functions; empty/informational state is not proof of no inlining. | ec9474724091 |
| 30 | Bottom tab: Source Markers | long_d128-tq128-tk64-lat0-source-bottom-source-markers.png | Lower Source pane selected to Source Markers; no displayed markers is not proof of no source events. | 585ec99a7981 |
| 31 | Bottom tab: Statistics | long_d128-tq128-tk64-lat0-source-bottom-statistics.png | Lower Source pane selected to Statistics for the current source/disassembly context. | b8acaa0a0a91 |
| 32 | Raw metrics | long_d128-tq128-tk64-lat0-raw.png | Raw native-report GUI page; raw.csv is the machine-readable cross-check. | e1a9fa502b85 |
| 33 | PTX supplement: left code | long_d128-tq128-tk64-lat0-source-ptx-horizontal-left-code.png | Additional horizontal view of the PTX-mode left code/source pane. | 6164054d6e12 |
| 34 | PTX supplement: left metrics | long_d128-tq128-tk64-lat0-source-ptx-horizontal-left-metrics.png | Additional horizontal view of the PTX-mode left metrics pane. | af5f58f3fd61 |
| 35 | PTX supplement: right code | long_d128-tq128-tk64-lat0-source-ptx-horizontal-right-code.png | Additional horizontal view of the PTX-mode right code pane. The SHA-256 comparison found no visible change from its paired right-side image. | b403c751c1ba |
| 36 | PTX supplement: right metrics | long_d128-tq128-tk64-lat0-source-ptx-horizontal-right-metrics.png | Byte-identical to the right-code capture in every report; no distinct scroll is proven. The SHA-256 comparison found no visible change from its paired right-side image. | b403c751c1ba |

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

**Review status:** all 36 screenshots were opened and visually inspected individually at 1936×1048.

### Summary and Details captures

1. **Summary** — result 851 / kernel; grid (1,16,8), block (128,1,1); duration 343.78 µs; 8,547,776 cycles; RTX 4060 at 2.49 GHz; process [9532] python.exe; compute capability 8.9, 24 SMs. Row: estimated speedup 72.48%, runtime improvement 249.17 µs, compute throughput 22.98%, memory throughput 27.52%, 255 registers/thread.
2. **Details 01** — compute 22.98%, memory 27.52%, L1/TEX 21.98%, L2 27.52%, DRAM 13.98%; duration 343.78 µs, elapsed cycles 8,547,776, SM active cycles 7,532,274.58; SM/DRAM frequency 2.49/8.24 GHz. Memory throughput 36.88 GB/s; L1/TEX hit 8.88%; L2 hit 91.21%; compression counters 0; persisting L2 4.72 MiB; sector-promotion misses 0; Mem Busy 27.35%, Max Bandwidth 27.52%, Mem Pipes Busy 13.03%; local/shared spill requests and overhead 0. Scheduler active 1.00, eligible/issued 0.15; No Eligible 85.22%, One or More Eligible 14.78%; Issue Slot Utilization 72.48% estimate, with one instruction issued about every 6.8 cycles.
3. **Details 02** — repeated memory and scheduler rows; warp cycles per issued/executed instruction 6.77; average active threads/warp 32.00; non-predicated-off threads/warp 31.69. Optional smsp__pcsamp_sample_count not found. Math Pipe Throttle rule begins with estimate 33.70%.
4. **Details 03** — Math Pipe Throttle explanation attributes about 2.3 cycles/warp to waiting for execution-pipe availability. This accounts for the 33.70% NCU estimate of the 6.8 cycles between issues; it is an NCU heuristic, not a measurement of achievable speedup.
5. **Details 04** — launch grid 128, registers/thread 255, block 128, threads 16,384, waves/SM 5.33, Green Context 0, 24 SMs, all TPC IDs. Static shared 98.30 KiB/block; dynamic shared 0; driver shared 1.02 KiB; configuration 102.40 KiB; stack 1,024 bytes; 12 TPCs.
6. **Details 05** — repeats Scheduler and Warp State plus launch details; occupancy begins at the bottom. Shared memory remains the likely CTA residency cap.
7. **Details 06** — theoretical occupancy 8.33%, theoretical active warps/SM 4, achieved occupancy 8.35%, achieved active warps/SM 4.01. CTA limits: registers 2, shared memory 1, warps 12, SM 24. Theoretical Occupancy rule points to shared-memory demand.
8. **Details 07** — GPU/Memory Distribution begins. Average SM and L1 active cycles 7,532,274.58; average L2 active cycles 5,997,756.17; average DRAM active cycles 3,962,200; average SMSP active cycles 7,507,755.78.
9. **Details 08** — distribution and imbalance tail: SM estimate 10.12%, maximum instance 11.49% above average, minimum 11.04% below; SMSP estimate 9.96%, maximum 11.35% above average, minimum 11.11% below.
10. **Details 09** — final view of the same launch/occupancy/distribution area; the imbalance values are the same as Details 08 and are overlapping scroll evidence, not another run.

### Source, instruction and pane captures

11. **Source 01** — Source tab, left Source file _stub.py, right SASS source kernel, Navigate By Instructions Executed. Left pane shows ScalarProtocol wrappers; right SASS begins with register, thread-ID and CTA-ID setup. Lower tab is Instructions & Dependencies / Scoreboard Dependencies with empty self/input/output tables.
12. **Source 02** — left _stub.py around lines 81–103, with __rfloordiv__, __rmod__, __rpow__, __rand__, __ror__, __rxor__ wrappers. SASS remains at the initial setup instructions.
13. **Source 03** — around lines 90–113: __rpow__, __rand__, __ror__, __rxor__, __ge__, __gt__. Same leading SASS rows.
14. **Source 04** — around lines 99–122: __ror__, __rxor__, __ge__, __gt__, __le__, __lt__. SASS remains at the initial listing.
15. **Source 05** — around lines 108–131: __ge__, __gt__, __le__, __lt__, __eq__, __ne__. The source is generated Tile scalar protocol code, not the attention implementation.
16. **Source high-level head** — _stub.py around lines 63–86, with __xor__, __radd__, __rsub__, __rmul__, __rtruediv__, __rfloordiv__ declarations and ellipsis bodies.
17. **Source high-level tail** — around lines 72–95: __rsub__, __rmul__, __rtruediv__, __rfloordiv__, __rmod__, __rpow__. These are API/stub methods.
18. **SASS selected-line** — a visible blue selection is around source line 78 rather than a literal line 91. The right SASS table still shows the kernel's initial launch/index sequence. The filename suffix line91 is not a reliable description of the highlighted line.
19. **Horizontal left-code** — left code has been shifted horizontally; leading parts of method declarations are clipped and the right ends containing TileOrScalar remain.
20. **Horizontal left-metrics** — Instructions Executed, Attributed Stalls and Attributed Live Registers headings are exposed; no visible values are populated.
21. **Horizontal right-code attempt** — SASS and the rightmost Instruction Category / Instructions Executed columns are visible. The old right-code and right-metrics captures are identical.
22. **Horizontal right-metrics attempt** — byte-identical to item 21; the original capture set therefore did not demonstrate independent right subpanel movement.
23. **Source view picker** — right view selected as SASS; horizontal pane boundaries and the instruction-category bars are visible.
24. **PTX view** — Right View PTX; source selector _stub.py,cutile_attention.py. Header shows .version 9.4, .target sm_89, .address_size 64, launch metadata symbol, the generated kernel entry and long parameter declarations, with the right side clipped.
25. **PTX selected-line** — still shows PTX header and kernel parameters rather than the inner kernel body; left source remains _stub.py around lines 72–95.
26. **Initial PTX horizontal code** — parameter signature remains too wide for the pane; only a clipped stretch of the generated entry/parameters is visible.
27. **Initial PTX horizontal metrics** — attribution headers on the left; PTX source/parameter rows at right.
28. **Instructions & Dependencies** — right pane is now PTX/Tile IR around lines 5584–5607. Visible sequence includes reduce_lhs/reduce_rhs, addf and yield, reshape_128, fma, ftof, broadcast, mulf, mmaf, continue, another broadcast, divf, reshape, constant<i32 0>, ftof, make_partition_view, store_view_tko, return and source_end. Type operands include 128-sized tile shapes and are partially clipped.
29. **Inline Functions** — same Tile IR tail remains in the right source pane; lower message states “Select a source line in an active inline function to show additional information.”
30. **Source Markers** — same PTX Tile IR region; filter All; columns File, Line/Address, Marker, Source; no rows.
31. **Statistics** — same Tile IR region; Aggregate/Maximum/Minimum/Average and Live Registers/Instruction Mix/Instructions Executed/Attributed Stalls/Attributed Live Registers headers; no values.
32. **Raw** — ID 0; 2026-Oct-31 19:25:03; API Call ID 851; estimated speedup 72.48%; runtime improvement 249.17 µs; 9 issues; kernel; process [9532] python.exe; thread 14696; RTX 4060; grid (1,16,8), block (128,1,1), 3 dimensions; 8,547,776 cycles; GPU duration 343.78 µs; compute 22.98%, memory 27.52%; spill requests 0. Mangled/demangled names and some addresses are clipped.
33. **PTX supplement — left code** — source begins with NVIDIA copyright, Apache-2.0 SPDX, future annotations, functools/inspect/textwrap/dataclass/typing and CUDA Tile imports, then Types/ScalarProtocol.
34. **PTX supplement — left metrics** — same source start with the Instructions Executed, Attributed Stalls and Attributed Live Registers columns visible.
35. **PTX supplement — right code** — horizontally shifted kernel signature/parameter list; the visible fragment starts inside the generated symbol/parameter string.
36. **PTX supplement — right metrics** — a second shifted crop of the same long PTX parameter declarations, with a different horizontal offset.

### Visible SASS instruction transcription

The 23 instruction rows visible in the leading SASS pane (operands cut off by the pane are explicitly truncated):

1. MOV R1, c[0x0][0x28]
2. S2R R4, SR_TID.X
3. MOV R0, c[0x0][0x28]
4. S2UR UR10, SR_CTAID.X
5. ULDC.64 UR8, c[0x0][0x118]
6. S2R R6, SR_CTAID.Z
7. IADD3 R3, P0, R0, c[0x0][0x170], …
8. CS2R R28, SRZ
9. IADD3 R0, P2, R0, c[0x0][0x174], …
10. S2R R69, SR_CTAID.Y
11. IADD3.X R8, RZ, RZ, RZ, P0, …, I PT (truncated)
12. CS2R R30, SRZ
13. LOP3.LUT P1, RZ, R0, 0xffffffff, …
14. CS2R R24, SRZ
15. SHF.R.U64 R3, R3, 0x7, R8
16. CS2R R26, SRZ
17. IADD3 R1, R1, -0x156, RZ
18. CS2R R12, SRZ
19. CS2R R14, SRZ
20. CS2R R10, SRZ
21. CS2R R32, SRZ
22. CS2R R34, SRZ
23. CS2R R36, SRZ

Instruction-category panel shows Movement, Miscellaneous, Integer and Uniform Data, with the displayed execution percentages below 0.01%.

### Per-configuration interpretation

At 343.78 µs, this TQ128/TK64 report has 85.22% No Eligible cycles and about 0.15 eligible warps/scheduler. The 98.30 KiB shared-memory block footprint limits residency to one CTA/SM and approximately 8.33% theoretical occupancy. The NCU Math Pipe Throttle diagnostic reports about 2.3 cycles/warp waiting for execution-pipe availability, but its 33.70% estimated speedup is heuristic. DRAM throughput is 13.98%, not a saturated-bandwidth signature. No local/shared spill requests are reported. The source attribution is generated _stub.py, and the visible SASS is setup/index arithmetic rather than the attention loop.
