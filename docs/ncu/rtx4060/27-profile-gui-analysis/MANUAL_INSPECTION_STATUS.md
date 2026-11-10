# Manual inspection status — 27 NCU configurations

**Updated:** 2 November 2026  
**Profiles:** 27  
**Original GUI screenshots:** 972 (864 primary + 108 PTX supplement)  
**Per-profile analysis:** 27 Markdown files under `profiles/`

## What has been checked

- All 27 per-profile contact sheets were visually opened and checked for coverage. Each sheet contains 36 thumbnails: 32 primary views and four PTX supplement views.
- The contact-sheet pass checked the expected families: Summary, nine Details scroll positions, five Source scroll positions, Source high-level head/tail, selected-line SASS/PTX, left/right horizontal pane captures, Source view-picker/mode views, four lower Source tabs and Raw metrics.
- Representative full-resolution Summary, Details and Source screenshots were inspected, including a medium-D128 latency-10 profile's summary, scheduler/warp-state details and Source/SASS correlation.
- The original NCU metric exports and available per-profile PTX/SASS source-correlation exports were checked across all 27 profiles.
- Each configuration already has its own Markdown analysis with metrics, interpretation, instruction fingerprints and screenshot inventory.

## Findings from Source/assembly views

- All 27 available SASS exports contain visible `HMMA.16816.F32` tensor-core instruction records, along with FFMA/FADD/FMUL and integer/addressing instructions. The corresponding PTX exports show FMA, multiply, add, max/exp, control-flow and synchronization operations.
- In the representative selected-line view inspected, Tile IR line 91 is `get_tile_block_id` (program/CTA block-ID extraction). SASS maps it to CTA-ID register reads such as `S2UR`/`S2R`; PTX maps it to `mov.u32` from `%ctaid.x/y/z`. It is not the attention score or P×V inner loop.
- The GUI Source pane correlates generated stub code/Tile IR rather than providing complete line-by-line mapping to the original high-level attention source.
- The source-correlation exports are sparse: omitted regions are represented by `...`. They are not a complete emitted PTX/SASS listing.
- Blank dependency/inline/source-marker panes are not evidence that dependencies, inlining or source events are absent.

## What is not complete

The owner-requested task was to manually transcribe every instruction and every small visual detail from every full-resolution screenshot for all 27 configurations. **That exact task is not complete.** Contact sheets are a coverage review, not high-resolution transcriptions. Representative full-resolution views and per-profile instruction fingerprints do not substitute for reading and recording all details in all 972 images.

The [disassembly inventory](DISASSEMBLY_INVENTORY.md) is a cross-profile index of visible instruction families in the exported text. Its row counts count visible export records, not dynamic instruction executions or every instruction emitted by the compiler.

## Next manual-review work

1. Open every original full-resolution screenshot in each profile's 36-view sequence.
2. Transcribe visible values, labels, instruction text, column headings, selected lines, empty panels and pane positions; distinguish displayed values from interpretation and unreadable/omitted areas.
3. Cross-check each transcription against `details.csv`, `raw.csv`, `source-tileir-ptx.txt` and `source-tileir-sass.txt`, without treating those exports as a visual substitute.
4. Update each profile README only after its full image set has been inspected.
5. Reconcile all profile reports and this index before using them as final-report sources.
