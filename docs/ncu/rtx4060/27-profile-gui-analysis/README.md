# 27-profile Nsight Compute screenshot-by-screenshot analysis

**Every one of the 27 legal NCU experiments now has its own report** in profiles/, including per-case metrics, scheduler/resource interpretation, PTX/SASS fingerprints and a 36-image screenshot-by-screenshot index.

## Capture provenance

**Update, 2 November 2026:** the old right-pane screenshots remain invalid as movement evidence (the old right-code/right-metrics pairs are identical). A corrected crop-local probe passes on one representative report: all four crops changed during horizontal movement and restored exactly to baseline in both SASS and PTX modes ([run #41478616763](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41478616763)). This is **one profile only**, not a 27-profile closure. The all-profile attempt [#41478794871](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41478794871) has a failed capture step with publication still running; follow-up [#41479365336](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41479365336) remains queued. Do not mark the full matrix fixed.

- Native reports: [run 41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024), 27/27 PASS on Windows RTX 4060, SM 8.9, driver 616.92.
- Actual NCU GUI screenshot capture: [run 41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158), 864 PNGs (32/report), zero skipped.
- PTX supplement: [run 41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876), 108 PNGs (4/report), zero skipped.
- Total 972 genuine 1936Ã—1048 GUI screenshots, 36 per report. The captures imported existing reports sequentially; the screenshot stage did not replay or reprofile the GPU.
- The original two base D=128 experiments remain preserved separately in results/ncu/rtx4060/20261027-230742/ncu-analysis-current/README.md.

## Individual experiment reports
| Workload | Report |
|---|---|
| long_d128 | [ long_d128-tq128-tk128-lat0 ](profiles/long_d128-tq128-tk128-lat0.md) |
| long_d128 | [ long_d128-tq128-tk64-lat0 ](profiles/long_d128-tq128-tk64-lat0.md) |
| long_d128 | [ long_d128-tq32-tk64-lat0 ](profiles/long_d128-tq32-tk64-lat0.md) |
| long_d128 | [ long_d128-tq64-tk128-lat0 ](profiles/long_d128-tq64-tk128-lat0.md) |
| long_d128 | [ long_d128-tq64-tk64-lat0 ](profiles/long_d128-tq64-tk64-lat0.md) |
| longer_batched_d128 | [ longer_batched_d128-tq128-tk128-lat0 ](profiles/longer_batched_d128-tq128-tk128-lat0.md) |
| longer_batched_d128 | [ longer_batched_d128-tq128-tk64-lat0 ](profiles/longer_batched_d128-tq128-tk64-lat0.md) |
| longer_batched_d128 | [ longer_batched_d128-tq32-tk64-lat0 ](profiles/longer_batched_d128-tq32-tk64-lat0.md) |
| longer_batched_d128 | [ longer_batched_d128-tq64-tk64-lat0 ](profiles/longer_batched_d128-tq64-tk64-lat0.md) |
| medium_d128 | [ medium_d128-tq128-tk128-lat0 ](profiles/medium_d128-tq128-tk128-lat0.md) |
| medium_d128 | [ medium_d128-tq128-tk64-lat0 ](profiles/medium_d128-tq128-tk64-lat0.md) |
| medium_d128 | [ medium_d128-tq32-tk64-lat0 ](profiles/medium_d128-tq32-tk64-lat0.md) |
| medium_d128 | [ medium_d128-tq64-tk64-lat10 ](profiles/medium_d128-tq64-tk64-lat10.md) |
| medium_d64 | [ medium_d64-tq128-tk128-lat0 ](profiles/medium_d64-tq128-tk128-lat0.md) |
| medium_d64 | [ medium_d64-tq128-tk64-lat0 ](profiles/medium_d64-tq128-tk64-lat0.md) |
| medium_d64 | [ medium_d64-tq32-tk64-lat0 ](profiles/medium_d64-tq32-tk64-lat0.md) |
| medium_d64 | [ medium_d64-tq64-tk128-lat0 ](profiles/medium_d64-tq64-tk128-lat0.md) |
| medium_d64 | [ medium_d64-tq64-tk64-lat0 ](profiles/medium_d64-tq64-tk64-lat0.md) |
| short_d32 | [ short_d32-tq32-tk64-lat0 ](profiles/short_d32-tq32-tk64-lat0.md) |
| short_d32 | [ short_d32-tq64-tk64-lat0 ](profiles/short_d32-tq64-tk64-lat0.md) |
| short_d64 | [ short_d64-tq32-tk64-lat0 ](profiles/short_d64-tq32-tk64-lat0.md) |
| short_d64 | [ short_d64-tq64-tk64-lat0 ](profiles/short_d64-tq64-tk64-lat0.md) |
| short_multihead_d64 | [ short_multihead_d64-tq128-tk128-lat0 ](profiles/short_multihead_d64-tq128-tk128-lat0.md) |
| short_multihead_d64 | [ short_multihead_d64-tq128-tk64-lat0 ](profiles/short_multihead_d64-tq128-tk64-lat0.md) |
| short_multihead_d64 | [ short_multihead_d64-tq32-tk64-lat0 ](profiles/short_multihead_d64-tq32-tk64-lat0.md) |
| short_multihead_d64 | [ short_multihead_d64-tq64-tk128-lat0 ](profiles/short_multihead_d64-tq64-tk128-lat0.md) |
| short_multihead_d64 | [ short_multihead_d64-tq64-tk64-lat0 ](profiles/short_multihead_d64-tq64-tk64-lat0.md) |

## Matrix-wide findings
| Metric | Range / result | Interpretation |
|---|---:|---|
| NCU duration, normalized | 4â€“1,370 Âµs | Instrumented NCU values; original units retained on individual pages. |
| Compute throughput | 0.41%â€“28.98% | Low relative to device peak in all reports. |
| DRAM throughput | 5.79%â€“30.42% | DRAM not saturated. |
| Memory bandwidth | 15.14â€“79.76 GB/s | Workload-specific; max remains well below the device peak. |
| No Eligible cycles | 75.11%â€“90.78% | Schedulers often lacked ready warps; not whole-GPU idle time. |
| Achieved occupancy | 8.13%â€“15.52% | Constrained by per-CTA resource footprint and workload shape. |
| Static shared memory per CTA | 24.58â€“98.30 KiB | Large D=128 allocations produce one CTA/SM limits in most profiles. |
| Spilling | 0 local and 0 shared requests, all 27 | No spill evidence. |

### Findings
- D=128 cases generally allocate 65.54â€“98.30 KiB static shared memory per CTA, often limiting residency to one CTA/SM and 8.33% theoretical occupancy.
- Best scheduler/resource case in the 27-profile set: medium_d64-tq32-tk64-lat0 (15.52% achieved occupancy, 0.31 eligible warps/scheduler, 75.11% No Eligible, 213 registers/thread, 45.06 KiB shared memory). Follow-up candidate only; not a production-default decision by itself.
- Maximum DRAM throughput is 30.42% and maximum memory bandwidth is 79.76 GB/s; simple DRAM saturation is not a common explanation. Memory latency/scoreboard waits remain possible.
- All 27 profiles report zero local and shared spill requests.
- NCU rule-engine estimated speedups are diagnostic heuristics, not measured gains. NCU replay duration is not normal unprofiled CUDA-event latency.
- Short cases underfill the GPU and are sensitive to fixed launch/profile overhead; do not use their occupancy alone to predict long-workload scaling.

## Screenshot audit and outstanding limitation
- In the original 27-profile captures, the SASS right-code/right-metrics pairs are byte-identical, and the supplemental PTX right-code/right-metrics pairs are also byte-identical. Those original images do not prove right-side horizontal movement.
- The corrected one-profile probe (#41478616763) verified changed pixels in all four pane-local crops for both SASS and PTX and exact restoration after inverse scrolling. The corrected full 27-profile rerun is still open; see [capture audit](CAPTURE_AUDIT.md).
- Left-code/left-metrics pairs differ in the original evidence. Screenshot records and SHA-256 prefixes are indexed per profile.
- Blank dependency/inline/source-marker panes are not interpreted as zero dependencies, no inlining or no source events. Generated stub correlation is not full source mapping of the high-level kernel.

## Manual inspection and instruction inventory

All 27 per-configuration Markdown reports are present under `profiles/`. The 27 profile contact sheets (36 view thumbnails each) have been visually checked for coverage and obvious missing/blank views. Representative full-resolution Summary, Details and Source panes were inspected. This is not a literal manual transcription of every detail in all 972 full-resolution screenshots. See [manual inspection status](MANUAL_INSPECTION_STATUS.md).

The extracted Source-correlation text exports are sparse: they contain omitted `...` rows and generated-stub/Tile IR correlation rather than full source mapping to the high-level attention kernel. The [disassembly inventory](DISASSEMBLY_INVENTORY.md) is a navigation index of visible mnemonic families, not a complete instruction transcript.

## Artifact links
- [Native profile report/CSV/PTX/SASS artifact](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024)
- [864 primary screenshot artifact](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158)
- [108 PTX supplement artifact](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876)
- [Original two-base-experiment analysis](../../../../results/ncu/rtx4060/20261027-230742/ncu-analysis-current/README.md)