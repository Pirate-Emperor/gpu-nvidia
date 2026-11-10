# Visible PTX/SASS source-export inventory — 27 NCU profiles

Generated from the `source-tileir-sass.txt` and `source-tileir-ptx.txt` exports belonging to native NCU run [#41206212024](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41206212024).

**Interpretation limit:** counts below are visible instruction records in exported text, not unique opcodes, dynamic executed instructions, or a complete machine-code listing. NCU exports omit regions using `...`; their counts are not a measure of work and should not be compared as performance metrics. The GUI Source pane uses generated stub/Tile IR correlation, not complete high-level Python source mapping. This inventory is a navigation aid, not the requested manual transcription of every screenshot.

All 27 SASS exports contain visible `HMMA.16816.F32` tensor-core instruction records. The individual native reports and their Details/Raw exports remain the authoritative profiler evidence.

| Profile | SASS visible rows | PTX visible rows | Frequent SASS mnemonics | Frequent PTX mnemonics |
|---|---:|---:|---|---|
| `long_d128-tq128-tk128-lat0` | 6,866 | 3,058 | FFMA 915; MOV 800; HMMA 512; FADD 404; FMUL 388 | fma 400; mul 388; add 272; shl 142; max 136 |
| `long_d128-tq128-tk64-lat0` | 4,896 | 2,001 | FFMA 787; MOV 643; FMUL 260; HMMA 256; FADD 212 | mul 260; fma 208; add 144; mov 129; div 128 |
| `long_d128-tq32-tk64-lat0` | 4,061 | 2,836 | FFMA 343; PRMT 304; FMUL 272; HMMA 256; FADD 256 | mov 310; mul 272; fma 256; add 176; add.s32 135 |
| `long_d128-tq64-tk128-lat0` | 3,416 | 1,896 | FFMA 461; HMMA 256; MOV 235; FADD 202; FMUL 194 | fma 200; mul 194; add 136; mov 97; shl 87 |
| `long_d128-tq64-tk64-lat0` | 2,537 | 1,290 | FFMA 397; MOV 158; FMUL 130; PRMT 128; HMMA 128 | mul 130; fma 104; mov 97; add 72; div 64 |
| `longer_batched_d128-tq128-tk128-lat0` | 6,852 | 3,084 | FFMA 915; MOV 808; HMMA 512; FADD 404; FMUL 388 | fma 400; mul 388; add 272; shl 142; max 136 |
| `longer_batched_d128-tq128-tk64-lat0` | 4,963 | 2,027 | FFMA 787; MOV 693; FMUL 260; HMMA 256; FADD 212 | mul 260; fma 208; add 144; mov 129; div 128 |
| `longer_batched_d128-tq32-tk64-lat0` | 4,055 | 2,879 | FFMA 343; PRMT 304; FMUL 272; HMMA 256; FADD 256 | mov 310; mul 272; fma 256; add 176; add.s32 135 |
| `longer_batched_d128-tq64-tk64-lat0` | 2,601 | 1,317 | FFMA 397; MOV 181; FMUL 130; PRMT 128; HMMA 128 | mul 130; fma 104; mov 97; add 72; div 64 |
| `medium_d128-tq128-tk128-lat0` | 6,866 | 3,058 | FFMA 915; MOV 800; HMMA 512; FADD 404; FMUL 388 | fma 400; mul 388; add 272; shl 142; max 136 |
| `medium_d128-tq128-tk64-lat0` | 4,896 | 2,001 | FFMA 787; MOV 643; FMUL 260; HMMA 256; FADD 212 | mul 260; fma 208; add 144; mov 129; div 128 |
| `medium_d128-tq32-tk64-lat0` | 4,061 | 2,836 | FFMA 343; PRMT 304; FMUL 272; HMMA 256; FADD 256 | mov 310; mul 272; fma 256; add 176; add.s32 135 |
| `medium_d128-tq64-tk64-lat10` | 4,522 | 2,834 | FFMA 537; FMUL 390; HMMA 384; FADD 318; MOV 296 | mul 390; fma 312; add 216; add.s32 127; shl 125 |
| `medium_d64-tq128-tk128-lat0` | 4,426 | 2,659 | FFMA 595; FADD 404; FMUL 324; MOV 292; HMMA 256 | fma 400; mul 324; add 272; shl 143; max 136 |
| `medium_d64-tq128-tk64-lat0` | 4,248 | 2,828 | FFMA 607; FADD 424; FMUL 392; MOV 285; HMMA 256 | fma 416; mul 392; add 288; shl 149; max 144 |
| `medium_d64-tq32-tk64-lat0` | 2,480 | 1,889 | FFMA 219; PRMT 208; FADD 192; FMUL 156; IMAD 117 | mov 205; fma 192; mul 156; add 132; add.s32 87 |
| `medium_d64-tq64-tk128-lat0` | 2,411 | 1,557 | FFMA 301; FADD 202; FMUL 162; MOV 159; HMMA 128 | fma 200; mul 162; add 136; shl 77; add.s32 76 |
| `medium_d64-tq64-tk64-lat0` | 3,265 | 2,310 | FFMA 377; FADD 318; FMUL 294; MOV 217; HMMA 192 | fma 312; mul 294; add 216; shl 117; max 108 |
| `short_d32-tq32-tk64-lat0` | 3,116 | 2,651 | FADD 320; FFMA 267; FMUL 220; LOP3 143; FMNMX 120 | fma 320; mul 220; add 220; add.s32 153; mov 126 |
| `short_d32-tq64-tk64-lat0` | 3,212 | 2,606 | FADD 424; FFMA 367; FMUL 328; MOV 160; SHF 151 | fma 416; mul 328; add 288; shl 150; max 144 |
| `short_d64-tq32-tk64-lat0` | 2,481 | 1,886 | FFMA 219; PRMT 208; FADD 192; FMUL 156; IMAD 122 | mov 205; fma 192; mul 156; add 132; add.s32 87 |
| `short_d64-tq64-tk64-lat0` | 3,248 | 2,322 | FFMA 377; FADD 318; FMUL 294; MOV 209; HMMA 192 | fma 312; mul 294; add 216; shl 117; max 108 |
| `short_multihead_d64-tq128-tk128-lat0` | 4,426 | 2,659 | FFMA 595; FADD 404; FMUL 324; MOV 292; HMMA 256 | fma 400; mul 324; add 272; shl 143; max 136 |
| `short_multihead_d64-tq128-tk64-lat0` | 4,248 | 2,828 | FFMA 607; FADD 424; FMUL 392; MOV 285; HMMA 256 | fma 416; mul 392; add 288; shl 149; max 144 |
| `short_multihead_d64-tq32-tk64-lat0` | 2,480 | 1,889 | FFMA 219; PRMT 208; FADD 192; FMUL 156; IMAD 117 | mov 205; fma 192; mul 156; add 132; add.s32 87 |
| `short_multihead_d64-tq64-tk128-lat0` | 2,411 | 1,557 | FFMA 301; FADD 202; FMUL 162; MOV 159; HMMA 128 | fma 200; mul 162; add 136; shl 77; add.s32 76 |
| `short_multihead_d64-tq64-tk64-lat0` | 3,265 | 2,310 | FFMA 377; FADD 318; FMUL 294; MOV 217; HMMA 192 | fma 312; mul 294; add 216; shl 117; max 108 |

In the representative selected-line view, Tile IR line 91 is `get_tile_block_id`: SASS shows CTA-ID register reads (for example `S2UR`/`S2R`) and PTX shows `mov.u32` from `%ctaid.x/y/z`. This is program/CTA index setup, not the attention inner loop.
