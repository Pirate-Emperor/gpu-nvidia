# NCU screenshot evidence audit

**Original screenshot evidence**
- Primary capture: workflow [#41326582158](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41326582158); 27 reports × 32 = 864 PNGs.
- PTX supplement: workflow [#41329271876](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41329271876); 27 reports × 4 = 108 PNGs.
- Total: **972 original GUI screenshots**, 36 per profile, each 1936×1048.
- Original SASS right-code/right-metrics pairs are byte-identical in all 27 reports; PTX right-code/right-metrics supplement pairs are also byte-identical. The original images therefore did not prove right-side horizontal scrolling.
- Left-code/left-metrics captures are distinct. The limitation was documented rather than hidden.

**Corrected probe status — 2 November 2026**
- [Run #41478616763](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41478616763) passed on one representative report (`long_d128-tq128-tk128-lat0`).
- All four pane-local crops (left code, left metrics, right code, right metrics) changed during horizontal movement in both SASS and PTX modes.
- Each inverse scroll restored the respective crop exactly to baseline (zero changed-pixel fraction); no WM_MOUSEHWHEEL fallback was required.
- This is a one-profile proof only. Do not generalise it to all 27.
- Full-matrix attempt [#41478794871](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41478794871) has a failed capture step with publication still running. Follow-up [#41479365336](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41479365336) remains queued. Full 27-profile corrected pane evidence is **not complete**.

| Profile | Primary | PTX supplement | Total | Original SASS right duplicate | Original PTX right duplicate |
|---|---:|---:|---:|---|---|
| `long_d128-tq128-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `long_d128-tq128-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `long_d128-tq32-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `long_d128-tq64-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `long_d128-tq64-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `longer_batched_d128-tq128-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `longer_batched_d128-tq128-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `longer_batched_d128-tq32-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `longer_batched_d128-tq64-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d128-tq128-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d128-tq128-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d128-tq32-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d128-tq64-tk64-lat10` | 32 | 4 | 36 | yes | yes |
| `medium_d64-tq128-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d64-tq128-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d64-tq32-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d64-tq64-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `medium_d64-tq64-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `short_d32-tq32-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `short_d32-tq64-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `short_d64-tq32-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `short_d64-tq64-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `short_multihead_d64-tq128-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `short_multihead_d64-tq128-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `short_multihead_d64-tq32-tk64-lat0` | 32 | 4 | 36 | yes | yes |
| `short_multihead_d64-tq64-tk128-lat0` | 32 | 4 | 36 | yes | yes |
| `short_multihead_d64-tq64-tk64-lat0` | 32 | 4 | 36 | yes | yes |

## Manual inspection boundary

All 27 per-profile contact sheets (36 views per profile) have been visually checked for coverage and obvious missing/blank views. Representative full-resolution Summary, Details and Source screenshots were inspected. The per-profile Markdown reports contain metric interpretation, instruction fingerprints and screenshot inventories.

The requested literal transcription of every instruction and every small visual detail from all 972 full-resolution images has **not** been completed. See [MANUAL_INSPECTION_STATUS.md](MANUAL_INSPECTION_STATUS.md).
