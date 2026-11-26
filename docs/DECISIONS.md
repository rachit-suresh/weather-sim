# Design decisions

Each decision lists the options, what each one costs and gives, our current lean, and how we
will settle it (usually: an experiment on the local leaderboard). Status: **open** unless marked.

Quick reminder of what is scored (see `docs/GRADING_NOTES.md`):
RMSE skill 30, heavy-rain CSI 15, anti-blur 10, shock causality 15, repo scan 15, brief 15.

---

## A. Model family

See `docs/MODEL_FAMILIES.md` for every family. Short version:

| Option | Pro | Con |
|---|---|---|
| Local CNN | Exactly local, small, fast, fits small data | Sees few cells per step |
| U-Net | Best RMSE of small models | Can leak shocks far away on day 1 |
| Physics-hybrid (advection + CNN) | Physical spreading, sharp, best story | More code |
| LightGBM per cell | Strong on small data, CPU friendly | Slow rollout feature building |
| ConvLSTM, GNN, windowed ViT | Worth one run each | Unlikely to beat CNN on 6k samples |
| FNO / global transformer | Fast | Breaks causality by design (ruled out) |
| Diffusion / GAN | Sharp rain | Slow on CPU, random, hurts RMSE |

**Lean:** CNN and physics-hybrid as main candidates, U-Net as RMSE candidate.
**Settle by:** one baseline per family, same harness, full leaderboard.

## B. What the network predicts

| Option | Meaning | Pro | Con |
|---|---|---|---|
| Full state | Predict tomorrow's values directly | Simple | Must re-learn the whole field every step; drifts |
| Change from today (residual) | Predict `tomorrow - today` | Easy start (zero = persistence), stable | At long leads, persistence is a bad anchor |
| Anomaly from climatology | Work on `x - clim(doy)` | Removes the seasonal cycle, model focuses on weather; natural decay toward clim at long leads | Needs climatology shipped (allowed: one-year file) |
| Residual on a baseline | Predict a correction to damped persistence | Starts at skill ~0.11 for free | Correction can be lazy (stays near the baseline) |

**Lean:** anomaly space with a residual output (`anom_tomorrow = anom_today + net(...)`).
**Settle by:** ablation on the CNN.

## C. How the 10 days are produced

| Option | Meaning | Pro | Con |
|---|---|---|---|
| Autoregressive 1-day step | Same network applied 10 times | Physically sensible, shock spreads naturally day by day, one model | Errors compound; needs multi-step training |
| Direct per lead | Separate output for each of the 10 days | No compounding | Ten times the output; no step-by-step physics; shock behaviour is hard to control |
| Hybrid | Autoregressive, then blend toward a direct or climatological forecast at long leads | Best of both | More moving parts |

Training choice inside autoregressive: how many steps to unroll in the loss (1, then 2, 4, 8 as a
curriculum). Longer unrolls cut drift but cost memory and time.

**Lean:** autoregressive with an unroll curriculum, plus a per-lead blend toward climatology.

## D. Rain handling

Rain is skewed: 8% of cells are exactly 0, the median is small, the max is 250 mm/day.

| Option | Pro | Con |
|---|---|---|
| Train on raw mm/day | Matches the RMSE metric | Heavy rain dominates the loss; light rain is ignored |
| log1p or sqrt transform | Balanced learning across light and heavy rain | Back-transform tends to under-predict extremes (hurts CSI) |
| Non-negative output (ReLU / softplus) | Never negative, guaranteed | ReLU can get stuck at 0 |
| Two-part head (rain yes/no + amount) | Good CSI, clean dry areas | More code |
| Extra loss weight above 10 mm | Targets CSI directly | Can over-forecast rain (false alarms reduce CSI too) |

Always: clamp to `>= 0` after the model (validity gate).
**Lean:** sqrt or log1p transform + non-negative output + heavy-rain weighting; test the two-part head later.

## E. Blur control

Plain MSE training gives the "average future", which is blurry. The RMSE points cap at skill 0.12,
so beyond that, sharpness is worth more than lower RMSE.

| Option | Pro | Con |
|---|---|---|
| Plain MSE | Best RMSE | Fails anti-blur for rain and wind at day 8 |
| MSE + spectral / variance loss | Keeps detail during training | Extra tuning |
| CRPS with a small ensemble | Sharp members, honest spread | Costs 2-4x training compute |
| Post-processing (variance rescale or quantile mapping) | Cheap, works on any model | Can add noise-like detail; must stay under 2.5x real variance |
| Advection (family 10) | Sharp by construction | Only for the hybrid family |

**Lean:** target skill ~0.15-0.2 for safety margin, then add a variance loss and per-channel
post-scaling to pass anti-blur.

## F. Grid edges (limited-area problem)

We never see weather outside the box, but it flows in. Edge errors reach the centre in 3-4 days.

| Option | Pro | Con |
|---|---|---|
| Zero padding | Simple | Fake "walls" at the edge |
| Replicate / reflect padding | Smooth edges | Still invents inflow |
| Relax edge cells toward climatology | Realistic inflow at long leads; stable | Damps real systems entering from the edge |
| Edge-distance input channel | Lets the network learn edge behaviour | Uses capacity |

Note: the Arabian Sea shock box touches the western edge, so the edge choice affects one of the
two causality tests directly.
**Lean:** replicate padding + edge-distance channel + light climatology relaxation in a 2-cell band.

## G. Causality and determinism rules (fixed, not open)

- Per-step receptive field must stay below about 8 cells (the "no teleport" radius on day 1 is 11).
- No global pooling, no global attention, no Fourier layers, no normalisation over the whole grid
  (layer norm or instance norm across space). Batch norm is fine only in eval mode.
- No random numbers at inference, unless from a fixed seed reset at every call.
- Zeros as perturbation must give the same result as `null` (the engine always adds the
  perturbation the same way).

## H. Inference runtime

| Option | Pro | Con |
|---|---|---|
| Native torch | No conversion step | Sandbox torch version unknown; pickles can break across versions |
| ONNX (onnxruntime) | Version-robust, fast on CPU, available in sandbox | Export step; some ops do not export |
| Plain numpy | Zero dependency risk | Must reimplement the network forward pass |

Always: load at import time, `torch.set_num_threads(4)`, save `state_dict` (never whole pickled
models), measure total time including JSON.
**Lean:** torch `state_dict` during development, ONNX export for the final submission with a
torch fallback.

## I. Validation protocol

- Split by year: train 2005-2017, validate 2018-2019, test 2020-2021 (test touched rarely).
- Local grader (`training/evaluate.py`) reuses the kit's own `monsoon_r1.scoring` code.
- Event windows like the hidden ones, picked automatically from held-out years: monsoon onset,
  heavy rain, heatwave, cyclone, winter disturbance.
- Every run writes one row to `runs/leaderboard.csv` with all score components and speed.
- Never use 2023-2024 data for training (that is the hidden test period).

## J. Dashboard and impact extension

| Option | Pro | Con |
|---|---|---|
| Plain HTML/JS dashboard (chosen) | Repo scan sees it as frontend, never as inference; no Python UI libs | Hand-written charts |
| Streamlit / Plotly Dash | Fast to build | A Python file calling `/simulate` becomes "inference code" in the repo scan; importing plotly/streamlit there costs 2 points |

**Status: decided** -> HTML/JS in `dashboard/`, talking to `serve.py` over HTTP (CORS enabled).

Impact extension candidates (open): heatwave risk (t2m + humidity), crop stress under a stalled
monsoon (rain deficit), flood risk (multi-day rain totals), solar/wind generation (wind speed,
cloud proxy from tcwv/rain), ensemble uncertainty map. All must be computed from live
`/simulate` output.
