# Run 2: spec v1 recipe on the top 5 families

Date: 2026-11-26. Spec: `docs/SPEC.md` v1.0, section 12. Validation years 2018-2019.
Recipe (all families): sqrt rain (`REP-4`), heavy-rain weight 3x (`RAIN-2`), spread + spectral loss
(`SHP-1/2`), unroll curriculum 1:4, 2:3, 4:3, 8:2 (`ROL-2`). Then post-processing fitted on training years:
blend (`ROL-3`), rescale (`SHP-3`), rain quantile map (`SHP-4`).

Scores = judge's automatic parts out of 70. Events = 5 auto-picked extremes, random = 60 windows,
named = real events (Kerala floods 2018, cyclones Titli/Gaja/Fani/Vayu, onset 2019, heat 2019, `VAL-2`).

## Raw models (no post-processing)

| Family | Sweep 1 events / random | Run 2 events | Run 2 random | Run 2 named | Mean of 3 | Heavy-rain skill | Shocks /15 |
|---|---|---|---|---|---|---|---|
| U-Net | 60.9 / 64.4 | 63.4 | **68.0** | 68.1 | **66.5** | 0.071 | 13.1 |
| CNN + transformer hybrid | 59.9 / 63.5 | **64.0** | 66.4 | **68.5** | 66.3 | 0.072 | **13.5** |
| ConvGRU (time series) | 58.2 / 61.4 | 63.4 | 67.3 | 68.1 | 66.3 | 0.070 | 13.1 |
| CNN | 62.4 / 62.1 | 62.9 | 65.9 | 67.8 | 65.5 | 0.071 | 12.75 |
| Physics hybrid | 59.7 / 60.0 | 60.1 | 64.7 | 67.8 | 64.2 | 0.051 | 12.75 |

## Post-processing (effect vs raw, events / random)

| Family | + blend | + rescale | + rain map (forced on) |
|---|---|---|---|
| U-Net | +0.5 / -0.7 | no channel selected | +0.1 / -4.4 |
| Hybrid | -0.2 / -1.4 | no channel selected | +0.7 / -9.2 |
| ConvGRU | 0.0 / 0.0 | no channel selected | -9.1 / -8.1 |
| CNN | +0.4 / -0.2 | no channel selected | +0.7 / -6.1 |
| Physics | -0.3 / -1.4 | no channel selected | -4.7 / -13.7 |

Full table: `python -m training.report --prefix r2_`.

## What we learned

1. **The spec recipe helps every family.** +0.5 to +5.2 on event windows, +3.6 to +5.9 on random windows.
   The biggest winners are the hybrid (+4.1 events) and ConvGRU (+5.2 / +5.9). Heavy-rain skill rose from
   0.044-0.063 to about 0.071 for the top 4.
2. **U-Net, the CNN + transformer hybrid and ConvGRU are tied** (mean 66.3-66.5; the judge sees only 5 windows,
   so differences under ~2 points are noise). The CNN is about 1 point behind and the physics hybrid about 2 behind.
3. **Blend (`ROL-3`) does nothing useful:** between +0.5 and -1.4. It lowers error, but error points are
   already full, and it costs sharpness and shock response.
4. **Rescale (`SHP-3`) was never selected:** spread is already at 0.7-1.0 of real for every variable (the judge
   needs 0.4), so stretching never paid off.
5. **Rain quantile map (`SHP-4`) always hurts on random windows** (-4 to -14) even though it raises heavy-rain
   skill to 0.08-0.09: rain in the wrong place gets penalised twice, and shock responses get exaggerated
   (shocks drop to 9-11). The automatic rule kept it off for every family.
6. **Shocks:** the Arabian Sea wind shock fails "spreads outward" in 4-5 of 5 windows for EVERY family
   (about 1.9 points each). This is the edge problem (`EDG-2`). The sharper recipe also caused one
   "unstable" (> 5 std) warm-Bay response for CNN and U-Net. The hybrid and ConvGRU did not have it.
7. **Remaining points (best family):** heavy rain about 4 of 15 missing, shocks about 1.5-2 of 15 missing,
   blur about 0.3 missing. Error is full.

## Errors found (spec-driven classification)

| # | Type | What | Fix |
|---|---|---|---|
| 1 | **Spec error** | `ROL-3` applies the blend unconditionally, unlike `SHP-3`/`SHP-4` which are "keep only if validation improves". Run 2 shows it does not help. | v1.1: blend kept only if it improves the validation auto score; store `blend_on` with the fitted file |
| 2 | **Spec error** | `VAL-1` does not define the "named events" score. The code uses forecast + blur on named windows + shock points from the event windows. | v1.1: write that formula into `VAL-1` |
| 3 | **Code error** (fixed before running) | `scripts/resume_sweep.sh` quoted `$CUR`, which would have glued recipe flags into the curriculum value | split into `CUR` and `RECIPE` (commit 903e7e0) |
| 4 | **Code gap** (follows from #1) | `twin/models/post.py` turns the blend on whenever the spec does not force it | after v1.1: read `blend_on` from the fitted file |

## Proposed next steps (phase 2, needs spec v1.1)

1. Edge fix for the Arabian Sea shock (`EDG-2`: relax toward normal near edges F4, lat/lon inputs F6) on the top 3.
2. Rain: two-part head (`RAIN-3`, D4) on the top 3.
3. Stability guard: a training penalty or cap so a shock cannot change rain by more than 5 std.
4. CRPS small ensemble (`SHP-6`, E3) on one family; plain averaging ensemble of the top 3.
