# How the models work

Plain-language guide to what the code does, from the request coming in to the forecast going out,
and how each model family works inside. Measured numbers come from sweep 1 (`docs/RESULTS_SWEEP_1.md`).

---

## 1. The data

One day = a grid of 32 rows (latitude 6.5N..37.5N, row 0 = south) x 36 columns (longitude 65.5E..100.5E,
column 0 = west). Every cell has 6 numbers:

| # | Name | Meaning | Unit |
|---|---|---|---|
| 0 | `t2m` | temperature 2 m above ground | K |
| 1 | `tp` | rain over the whole day | mm/day |
| 2 | `u10` | wind toward the east, 10 m up | m/s |
| 3 | `v10` | wind toward the north, 10 m up | m/s |
| 4 | `msl` | air pressure at sea level | Pa |
| 5 | `tcwv` | water vapour in the whole air column above the cell | kg/m2 |

Plus 2 fixed maps that never change: land/sea and ground height.
A **map** in this doc = one 32 x 36 grid of numbers, one number per cell.

**Persistence** = "every future day equals the last input day". **Climatology** = "every future day equals
the 17-year average (2005-2021) for that calendar day, smoothed over +-7 days". These are the two free
forecasts the judge compares us against. Persistence wins at day 1; climatology wins from about day 4.

**Skill** = `1 - (our error / error of the better free forecast)`. 0 = no better than free, 0.12 = 12% smaller
error, negative = worse. The judge gives the full 30 error points at skill 0.12.

---

## 2. One request, step by step (`serve.py` -> `twin/engine.py`)

1. Read the request: 3 input days `[3, 6, 32, 36]`, the date as `[sin, cos]`, number of days, optional shock.
2. Add the shock to the last input day (the shock is a plain "+ this much" per cell and variable).
3. Clamp the inputs to safe ranges (rain >= 0, temperature -39.5..+59.5 C, ...) so a wild shock cannot break anything.
4. Turn `[sin, cos]` back into a day of the year (`atan2`).
5. Run the active model (`artifacts/active.json`) to get the 10 days.
6. Clamp the outputs again (no negative rain, temperature in range, no NaN).
7. If the model crashes, the damped-anomaly baseline answers instead (a crash would score 0 for that window).
8. Return JSON.

---

## 3. Anomalies (`twin/codec.py`): rewriting the numbers, not removing the season

`actual = normal for this date + anomaly`. The networks predict the **anomaly** (how unusual the day is),
then the server adds the normal back. Example: Delhi, 10 June: actual 308 K, normal 305 K -> anomaly +3 K.
The network predicts tomorrow's anomaly, say +2 K; the server returns 305.2 + 2 = 307.2 K.

The season is NOT lost. The network knows it three ways: the target day's normal values are an input,
the date is an input, and the output is normal + anomaly. Reasons for the rewrite:
- The predictable part ("July is hot") is handled exactly by the 17-year average; the network spends its
  capacity on the hard part ("why is THIS July unusual").
- Every variable gets a similar scale (about -3..+3), so pressure (~101,000) does not drown out wind (~5).
- Fading back to normal at long lead times is just "anomaly shrinks toward 0", easy to learn.
- Risk: the normal comes from 2005-2021; 2023-24 is warmer. To be measured and corrected (spec item).

Anomalies are divided by their typical size (standard deviation per variable, from the training years).

---

## 4. What a network sees for one forecast day (`twin/nets/common.py`)

The API gives 3 days x 6 variables = **18 maps**. We add **11 maps** built from things we already have,
so a network sees **29 maps** stacked (29 "channels", like a photo has 3 colour channels):

| Maps | Content | Where it comes from | Why |
|---|---|---|---|
| 1-6 | anomalies, 3 days ago | request | trend |
| 7-12 | anomalies, 2 days ago | request | trend |
| 13-18 | anomalies, last input day | request | current state, matters most |
| 19-24 | normal values for the day being predicted (scaled by global mean/std) | shipped climatology file | season and typical pattern, e.g. where the monsoon trough sits in July |
| 25 | land (1) / sea (0) | shipped static file | land heats fast, sea slowly |
| 26 | ground height in km | shipped static file | mountains force rain and block cold air |
| 27-28 | sin, cos of the target date | request date | season as a smooth clock (31 Dec and 1 Jan stay close) |
| 29 | distance to grid edge: 0 at the edge, 1 from 8 cells in | grid geometry | weather enters from outside the box unseen; lets the model be careful near edges |

Scaling choices keep every map roughly in -3..+3 (stable training): anomalies / their std, height in km,
edge distance capped at 8 cells (= the judge's 8 cells/day speed limit, about one day of travel).

---

## 5. The 10-day loop (`twin/nets/rollout.py`)

```
for each forecast day 1..10:
    build the 29 input maps
    network predicts the CHANGE of the anomaly for tomorrow
    tomorrow = today + change       (clamped to +-12 std so it can never explode)
    slide the 3-day window: drop the oldest day, add tomorrow
```
At forecast time it is always one day at a time, 10 times ("autoregressive"). The network's last layer
starts at zero, so an untrained network = "tomorrow = today"; training learns the difference.

---

## 6. Training (`training/train_nn.py`)

1. Split by year: train 2005-2017, validate 2018-2019, test 2020-2021 (kept for the end).
2. Convert all 6,209 days to anomalies once.
3. A sample = pick day t; inputs t-2..t; targets the next k days.
4. **Unroll curriculum.** Unrolling only exists in training. Unroll 1: the model gets true days, predicts
   1 day, is graded, and never sees its own mistakes. Unroll 4: it predicts day 1, gets its OWN day 1 back
   as input, predicts day 2, ... day 4, and all 4 are graded together, so it learns to cope with its own
   imperfect inputs (less drift over 10 days). Sweep 1: 5 epochs unroll 1, 3 epochs unroll 2, 3 epochs unroll 4.
5. Loss: squared error, each latitude row weighted by cos(latitude) (as the judge does).
6. AdamW optimiser, learning rate 0.001 with warm-up then decay, large updates clipped.
7. Keep the weights with the lowest validation error; also save the full training state every epoch
   (`runs/ckpt/`) so `--resume` continues after a shutdown.
8. Score with the local grader.

## 7. Local grader (`training/evaluate.py`)

Uses the organisers' own scoring code on held-out years:
- 5 automatically picked extreme windows (monsoon onset, heaviest monsoon rain, pre-monsoon heat,
  lowest sea pressure = cyclone, wettest winter day) and 60 random windows;
- the judge's two shocks + the zero shock on every event window;
- **reach probe**: a 1-cell shock in central India; `r95` = radius holding 95% of the response,
  `max` = farthest cell that moved at all (day 1 and day 4). Judge limit on day 1: 11 cells.

---

## 8. Building blocks

- **3x3 convolution:** a small stencil slides over the grid; the new value of a cell = learned weighted sum of
  the cell and its 8 neighbours, over all input maps; the same weights everywhere.
- **Reach (receptive field):** each 3x3 layer adds 1 cell. 7 layers -> 7 cells (9.9 cells diagonally).
  That is the theoretical reach; the effective reach is much smaller because far weights are tiny after
  training. Sweep 1 measured it: 95% of a shock's response stays within 1-3 cells on day 1 in every local model.
  Over 10 days reach adds up (day 10 can see the whole grid), which is physical and allowed.
- **GELU:** smooth "keep positives, squash negatives" between layers (without it, stacked layers collapse to one).
- **Residual:** a block computes `x + f(x)`, so it only adds a correction.
- **Replicate padding:** at the grid edge, missing neighbours are copies of the edge cell.
  v2 experiment E6 (spec `V2-E6`) adds the CNN option `pad="zeros"`: missing neighbours are 0, which in anomaly
  space means "normal weather outside the box", and `--random-crop` trains on random sub-windows (24x28 up to
  32x36, edge-distance map recomputed per crop) so edge behaviour is learned at many positions.

---

## 9. The model families

Measured on validation years. "auto" = judge's automatic parts out of 70 (events / random windows).
"reach d1" = farthest cell moved by a 1-cell shock on day 1.

### Baselines (`twin/models/baselines.py`) - no network
- Persistence: repeat the last day. Reach 0. Fails the qualifying gate.
- Climatology: the 17-year normal. Ignores input and shocks.
- **Damped anomaly:** `forecast = normal + a x (today - normal_today)`, `a` fitted per variable and lead day
  (`training/fit_damped.py`). Day 1 keeps ~82% of a temperature anomaly, day 8 ~21%. Reach 0, so shocks never
  spread. auto 43.7 / 42.3. Also the server's crash fallback.

### LightGBM (`twin/models/lgbm.py`, `training/train_lgbm.py`) - decision trees, not a network
Hundreds of small if/else trees, each fixing the previous ones' errors. Every cell becomes one table row of
77 numbers (its 3x3 neighbourhood on the last day, its own values on the 2 earlier days, normals, land, height,
date, edge). 6 tree models (one per variable) predict tomorrow's anomaly change. Reach exactly 1 cell/day.
Trained on 1-day steps only, so it drifts. auto 53.2 / 50.7, 590 ms per forecast.

### CNN (`twin/nets/cnn.py`)
Image-style network at full resolution: 1 input conv (29 -> 64 maps), 3 residual blocks of 2 convs, a 1x1
output conv (64 -> 6) starting at zero. 239k weights. Reach 7 cells/day (measured 9.9 = diagonal).
auto **62.4** / 62.1.

### U-Net (`twin/nets/unet.py`)
A CNN that also works at half resolution: 2 convs at 32x36 -> average to 16x18 -> 2 convs -> grow back ->
join with the full-size features -> 2 convs -> output. A conv at half size covers 2 real cells, so reach ~10
(measured 12.1, slightly past the 11-cell line but with < 5% of the energy). Only one shrink level on purpose:
deeper U-Nets see the whole map in one day and fail the shock tests. auto 60.9 / **64.4**.

### ConvGRU - CNN + time series (`twin/nets/convgru.py`)
A GRU is a memory cell from sequence models: it keeps a memory and updates it each step with two learned gates
(update gate: how much to overwrite; reset gate: how much old memory to use). Here the memory is a map
(64 x 32 x 36) and the gates are convolutions. Day 1: the 3 input days go into the memory one by one.
Later days: only the newest predicted day goes in, and the memory is carried across all 10 days, so it can
remember more than 3 days. Memory starts empty for every request (same input -> same output). Reach ~5/day
(measured 7.1). auto 58.2 / 61.4.

### Graph neural network (`twin/nets/gnn.py`)
Cells are nodes linked to their 8 neighbours. One round: every neighbour sends a message computed from
(my values, its values, its direction); each cell averages them and updates itself. Light version in the
sweep: width 48, 4 rounds = 4 hops/day (measured 5.4). On a regular grid it behaves like a CNN with extra
steps. auto 57.4 / 53.6.

### CNN + transformer hybrid (`twin/nets/attention.py`, `HybridCNNTransformer`)
Transformer = each cell makes a query ("what I look for"), key ("what I hold") and value ("what I give");
it mixes the values of other cells weighted by query-key match, so it learns WHICH neighbours matter
(e.g. upwind cells). To stay local, cells only attend inside 4x4 windows; every second block shifts the
windows by 2 so information crosses borders (Swin style). Layout: 2 convs -> 2 window-attention blocks -> conv
head. 160k weights. Reach ~9 (measured 10.4). auto 59.9 / 63.5.

### Window ViT (`twin/nets/attention.py`, `WindowViT`)
Pure transformer: 2x2 cell patches become tokens (16x18 tokens), 2 window-attention blocks, then each token is
turned back into its 2x2 cells. Reach ~12-13 (measured 14.2, past the 11-cell line). Transformers need much
more data than our 6k days; patches blur small features. auto 57.1 / 62.3.

### FNO, Fourier neural operator (`twin/nets/fno.py`) - control experiment
Turns each map into waves (Fourier transform), multiplies the 10x10 longest waves by learned numbers, turns
back. Every wave touches every cell, so a shock reaches the whole grid in one day, and the FFT treats the map as
wrapping around (east edge touching west edge). Measured: 95% of a 1-cell shock spreads over 10 cells on day 1
(others: 1-3), reaches the whole grid, and 3 of 10 shock runs went unstable. auto **63.4** / 61.2 (best rain
skill), but shocks 11.6/15. Not a candidate: a small live shock in the desk review would expose it.

### Physics hybrid (`twin/nets/physics.py`, full explainer in `docs/PHYSICS_HYBRID.md`)
Each day: (1) a small CNN predicts how far the air moved, per variable and cell, capped at 4 cells/day;
(2) every cell takes yesterday's value from its "departure point" (cell - movement) - this is semi-Lagrangian
advection, the method real weather models use, so patterns MOVE instead of smearing; (3) a second small CNN
adds heating, rain forming and fading. 75k weights. Reach ~8 (measured 7.1, the most physical).
auto 59.7 / 60.0.

### Ensembles (`twin/models/ensemble.py`)
Weighted average of several models, weights per variable and lead day (e.g. trust the CNN early, the normal
late). Optional "restore variance" rescales the averaged anomaly so the map keeps the members' spread
(averaging smooths maps, which costs blur points). Deterministic, so the zero-shock test still passes.

---

## 10. Reach summary (per forecast day)

| Model | Theoretical reach | Measured farthest, day 1 |
|---|---|---|
| Damped / persistence | 0 | 0 |
| LightGBM | 1 | 1.4 |
| Graph network (light) | 4 hops | 5.4 |
| ConvGRU | ~5 (9 on day 1) | 7.1 |
| Physics hybrid | ~8 | 7.1 |
| CNN | 7 (9.9 diagonal) | 9.9 |
| CNN + transformer | ~9 | 10.4 |
| U-Net | ~10 | 12.1 |
| ViT | ~12-13 | 14.2 |
| FNO | whole grid | 24 (whole grid) |

Judge's day-1 limit: less than 5% of the response beyond 11 cells.
