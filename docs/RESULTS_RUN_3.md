# Run 3, run 3b, ensembles and the submission choice

Date: 2026-11-26. Spec: `docs/SPEC.md` 1.6-1.8. Scores = judge's automatic parts out of 70
(error 30 + heavy rain 15 + blur 10 + shocks 15), organisers' scoring code. Mean = average of the three
window sets (5 extreme events, 60 ordinary days, named real events).

## Screening (phase 2a, U-Net + one option, validation 2018-19)

| Option | Events | Random | Named | Mean | Shocks | Verdict |
|---|---|---|---|---|---|---|
| none (r2_unet) | 63.4 | 68.0 | 68.1 | 66.5 | 13.1 | baseline |
| D4 rain head, soft/hard gate (spec 1.1) | 38.0 | 22.4 | 39.8 | 33.4 | 11.6 | spec error, fixed in 1.2 |
| **D4 rain head, straight-through (1.2)** | 64.5 | 68.3 | 68.9 | **67.2** | **13.9** | **wins** |
| edge relax | 66.9 | 65.9 | 68.1 | 67.0 | 13.1 | +0.47, just under the bar |
| lat/lon inputs | 66.1 | 66.4 | 68.1 | 66.9 | 13.1 | no |
| shock guard | 62.0 | 68.1 | 68.1 | 66.1 | 13.1 | no |
| **D4 + edge relax** (team-approved test) | 66.2 | 68.7 | 69.3 | **68.05** | **14.25** | **wins vs D4** |

## Run 3 (D4) and run 3b (D4 + edge relax), all families, validation 2018-19

| Family | Run 2 mean | Run 3 (D4) mean | Run 3b (D4 + edge) mean | Best |
|---|---|---|---|---|
| CNN | 65.5 | **67.9** | 67.6 | run 3 |
| U-Net | 66.5 | 67.2 (67.1 repeat) | **68.05** | run 3b |
| CNN + transformer hybrid | 66.3 | **68.0** | 66.5 | run 3 |
| ConvGRU | 66.3 | 66.2 | **67.8** | run 3b |
| Physics hybrid | 64.2 | **65.0** | 64.4 | run 3 |

- D4 helps every family except ConvGRU; edge relax helps U-Net and ConvGRU, hurts the others.
- Reproducibility: retraining U-Net + D4 with the same recipe gave 64.3/68.1/68.9 vs 64.5/68.3/68.9.
- Rain quantile map: always worse overall (-5 to -17 on ordinary days), kept off by `SHP-4`.

## Ensembles (phase 2c)

| Model | Events | Random | Named | Mean | Shocks |
|---|---|---|---|---|---|
| Mean of U-Net D4+edge, hybrid r3, CNN r3 | 67.4 | 68.1 | 68.1 | 67.9 | 13.1 |
| Same + restore variance | 67.8 | 68.1 | 68.1 | 68.0 | 13.1 |

The ensemble maxes every forecast part on ordinary and named days (55/55) but its shock score drops to 13.1.
Restore-variance rescales whole maps, so a 1-cell shock moved cells 18 cells away on day 1 (past the judge's
11-cell line, `LOC-1` risk). Not used.

## Final check on untouched test years 2020-21 (spec `DAT-2`)

| Candidate | Validation mean | Test mean | Test error skill (ordinary days) | Test heavy-rain skill (events / ordinary) |
|---|---|---|---|---|
| **CNN run 3 (`r3_cnn`)** | 67.9 | **68.0** | **0.118** | **0.108 / 0.105** |
| Hybrid run 3 | 68.0 | 67.0 | 0.104 | 0.100 / 0.089 |
| U-Net D4 + edge | 68.05 | 66.45 | 0.107 | 0.100 / 0.089 |
| ConvGRU run 3b | 67.8 | 64.55 | 0.104 | 0.075 / 0.081 |

**Submission model: `r3_cnn`** (`artifacts/active.json`). The only top model that keeps its score on new years,
simplest, 38 ms per forecast. Kit schema validator passes; a fresh clone without Git LFS serves it.

## Errors found in this phase

| Type | What | Fix |
|---|---|---|
| Spec error | `RAIN-3` soft gate in training, hard gate at inference: rain too high, rollout drift | spec 1.2, straight-through gate |
| Spec error | `ARC-3` did not cover a model that fails to LOAD (server would not start) | spec 1.7, engine falls back at load |
| Code error | `--options --edge-relax` parsed as a flag (screening jobs never started) | pass as `--options=...` |
| Operating mistake | editing `scripts/run3.sh` while it ran broke its last line | rule: never edit a running script |

## Open risks
- Hidden 2023-24 windows: warmer than training years; error skill on ordinary days is already near 0.12 on 2020-21.
- Weak-Arabian-wind shock barely changes rain; "spreads outward" fails in ~3/5 windows.
- Brief `BRIEF.pdf`: team name placeholder and the impact module description must match the final dashboard.
