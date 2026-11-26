// Dashboard (starter). Talks to the twin server over HTTP only; every forecast shown is live server output.
// Map orientation: row 0 of the grid is the SOUTH edge (lat 6.5), so we draw rows bottom-up.
const $ = id => document.getElementById(id);
const NLAT = 32, NLON = 36, CELL = 16;
const CH_NAMES = ["t2m", "tp", "u10", "v10", "msl", "tcwv"];

const S = {
  server: "", meta: null, windows: [], win: 0, ch: 0, day: 1, view: "base",
  base: null, shocked: null, shocks: [], drag: null,
};

// ---------- colour maps ----------
const MAPS = {
  heat: [[0, [49, 54, 149]], [0.25, [116, 173, 209]], [0.5, [255, 255, 191]], [0.75, [244, 109, 67]], [1, [165, 0, 38]]],
  rain: [[0, [250, 250, 250]], [0.2, [158, 202, 225]], [0.5, [33, 113, 181]], [0.8, [84, 39, 143]], [1, [63, 0, 125]]],
  div: [[0, [5, 48, 97]], [0.5, [247, 247, 247]], [1, [103, 0, 31]]],
  seq: [[0, [255, 255, 217]], [0.5, [65, 182, 196]], [1, [8, 29, 88]]],
};
const CH_MAP = ["heat", "rain", "div", "div", "seq", "seq"];

function colour(name, x) {
  const st = MAPS[name]; x = Math.min(1, Math.max(0, x));
  for (let k = 1; k < st.length; k++) {
    if (x <= st[k][0]) {
      const [x0, c0] = st[k - 1], [x1, c1] = st[k], f = (x - x0) / (x1 - x0 || 1);
      return c0.map((v, i) => Math.round(v + f * (c1[i] - v)));
    }
  }
  return st[st.length - 1][1];
}

// ---------- server ----------
async function api(path, body) {
  const r = await fetch(S.server + path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : {});
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status} ${await r.text()}`);
  return r.json();
}

async function connect() {
  S.server = $("server").value.replace(/\/$/, "");
  try {
    S.meta = await api("/meta");
    $("status").textContent = "connected: " + S.meta.model_name; $("status").className = "status ok";
  } catch (e) { $("status").textContent = "error: " + e.message; $("status").className = "status err"; }
}

function perturbationArray() {
  if (!S.shocks.length) return null;
  const last = S.windows[S.win].init_states[2];
  const p = Array.from({ length: 6 }, () => Array.from({ length: NLAT }, () => new Array(NLON).fill(0)));
  for (const s of S.shocks) {
    for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
      const lat = 6.5 + i, lon = 65.5 + j;
      if (lat < s.lat0 || lat > s.lat1 || lon < s.lon0 || lon > s.lon1) continue;
      p[s.ch][i][j] += s.kind === "percent" ? last[s.ch][i][j] * s.value / 100 : s.value;
    }
  }
  return p;
}

async function run() {
  if (!S.meta) await connect();
  const w = S.windows[S.win];
  const req = { init_states: w.init_states, doy_sincos: w.doy_sincos, steps: 10, perturbation: null };
  $("timing").textContent = "running...";
  try {
    const t0 = performance.now();
    const b = await api("/simulate", req);
    S.base = b.rollout;
    const pert = perturbationArray();
    S.shocked = pert ? (await api("/simulate", { ...req, perturbation: pert })).rollout : null;
    $("timing").textContent = `server ${b.inference_ms.toFixed(0)} ms, round trip ${(performance.now() - t0).toFixed(0)} ms`;
  } catch (e) { $("timing").textContent = "error: " + e.message; }
  draw(); IMPACT.render($("impact"), S.base, S.shocked, S.meta);
}

// ---------- drawing ----------
function field() {
  const w = S.windows[S.win];
  const pick = r => (S.day === 0 ? w.init_states[2] : r && r[S.day - 1]);
  if (S.view === "base") return pick(S.base)?.[S.ch];
  if (S.view === "shock") return pick(S.shocked || S.base)?.[S.ch];
  const a = pick(S.base), b = pick(S.shocked);
  if (!a || !b) return null;
  return a[S.ch].map((row, i) => row.map((v, j) => b[S.ch][i][j] - v));
}

function draw() {
  const cv = $("map"), g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  const f = field() || S.windows[S.win]?.init_states[2][S.ch];
  if (!f) return;
  let lo = Infinity, hi = -Infinity;
  for (const row of f) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  let cmap = CH_MAP[S.ch];
  if (S.view === "diff" || cmap === "div") { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; cmap = "div"; }
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const [r, gg, b] = colour(cmap, (f[i][j] - lo) / (hi - lo || 1));
    g.fillStyle = `rgb(${r},${gg},${b})`;
    g.fillRect(j * CELL, (NLAT - 1 - i) * CELL, CELL, CELL);
  }
  if (S.meta) drawCoast(g);
  for (const s of S.shocks) drawBox(g, s.lat0, s.lat1, s.lon0, s.lon1, "#00e5ff");
  if (S.drag) drawBox(g, ...dragBox(), "#fff");
  const unit = S.meta ? S.meta.units[S.ch] : "";
  $("legend").innerHTML = `<div style="display:flex;justify-content:space-between" class="hint"><span>${lo.toFixed(1)}</span>
    <span>${CH_NAMES[S.ch]} ${S.view === "diff" ? "difference " : ""}(${unit})</span><span>${hi.toFixed(1)}</span></div>
    <div style="height:10px;border-radius:3px;background:linear-gradient(90deg,${[0, .25, .5, .75, 1].map(x => `rgb(${colour(cmap, x)})`).join(",")})"></div>`;
}

function drawCoast(g) {
  const m = S.meta.land_sea_mask; g.strokeStyle = "rgba(0,0,0,.55)"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const land = m[i][j] > 0.5, x = j * CELL, y = (NLAT - 1 - i) * CELL;
    if (j + 1 < NLON && land !== (m[i][j + 1] > 0.5)) { g.moveTo(x + CELL, y); g.lineTo(x + CELL, y + CELL); }
    if (i + 1 < NLAT && land !== (m[i + 1][j] > 0.5)) { g.moveTo(x, y); g.lineTo(x + CELL, y); }
  }
  g.stroke();
}

function drawBox(g, lat0, lat1, lon0, lon1, col) {
  const x0 = (lon0 - 65) * CELL, x1 = (lon1 - 65) * CELL, y0 = (38 - lat1) * CELL, y1 = (38 - lat0) * CELL;
  g.strokeStyle = col; g.lineWidth = 2; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
}

// ---------- map interaction ----------
function cellAt(ev) {
  const r = $("map").getBoundingClientRect(), sx = $("map").width / r.width;
  const j = Math.floor((ev.clientX - r.left) * sx / CELL), i = NLAT - 1 - Math.floor((ev.clientY - r.top) * sx / CELL);
  return [Math.max(0, Math.min(NLAT - 1, i)), Math.max(0, Math.min(NLON - 1, j))];
}
function dragBox() {
  const [a, b] = S.drag;
  return [6 + Math.min(a[0], b[0]), 7 + Math.max(a[0], b[0]), 65 + Math.min(a[1], b[1]), 66 + Math.max(a[1], b[1])];
}

// ---------- shock list ----------
function renderShocks() {
  $("shocks").innerHTML = S.shocks.map((s, k) => `<div class="shock" data-k="${k}">
    <select data-f="ch">${CH_NAMES.map((c, i) => `<option value="${i}" ${i === s.ch ? "selected" : ""}>${c}</option>`).join("")}</select>
    <select data-f="kind"><option value="add" ${s.kind === "add" ? "selected" : ""}>+ value</option>
      <option value="percent" ${s.kind === "percent" ? "selected" : ""}>+ %</option></select>
    <input data-f="value" type="number" step="any" value="${s.value}">
    <br>lat <input data-f="lat0" type="number" value="${s.lat0}"> - <input data-f="lat1" type="number" value="${s.lat1}">
    lon <input data-f="lon0" type="number" value="${s.lon0}"> - <input data-f="lon1" type="number" value="${s.lon1}">
    <button data-del="${k}">x</button></div>`).join("");
}
const PRESETS = {   // same definitions as the judge (monsoon_r1.scoring.make_shock)
  warm_ocean: [{ ch: 0, kind: "add", value: 3, lat0: 8, lat1: 20, lon0: 82, lon1: 94 },
               { ch: 5, kind: "percent", value: 15, lat0: 8, lat1: 20, lon0: 82, lon1: 94 }],
  monsoon_stall: [{ ch: 2, kind: "percent", value: -40, lat0: 8, lat1: 20, lon0: 65, lon1: 75 },
                  { ch: 3, kind: "percent", value: -40, lat0: 8, lat1: 20, lon0: 65, lon1: 75 }],
};

// ---------- wiring ----------
async function init() {
  CH_NAMES.forEach((c, i) => $("channel").add(new Option(c, i)));
  try { S.windows = await (await fetch("data/demo_windows.json")).json(); }
  catch (e) { $("timing").textContent = "could not load data/demo_windows.json: serve this folder over http"; }
  S.windows.forEach((w, i) => $("window").add(new Option(w.label, i)));
  $("connect").onclick = connect;
  $("run").onclick = run;
  $("window").onchange = e => { S.win = +e.target.value; S.base = S.shocked = null; draw(); IMPACT.render($("impact"), null); };
  $("channel").onchange = e => { S.ch = +e.target.value; draw(); };
  $("day").oninput = e => { S.day = +e.target.value; $("dayLabel").textContent = S.day ? "+" + S.day : "input"; draw(); };
  $("view").onclick = e => { const v = e.target.dataset.v; if (!v) return; S.view = v;
    [...$("view").children].forEach(b => b.classList.toggle("on", b.dataset.v === v)); draw(); };
  $("presets").onclick = e => { const p = e.target.dataset.p; if (!p) return;
    S.shocks = p === "clear" ? [] : PRESETS[p].map(s => ({ ...s })); renderShocks(); draw(); };
  $("addShock").onclick = () => { S.shocks.push({ ch: 0, kind: "add", value: 2, lat0: 20, lat1: 25, lon0: 75, lon1: 80 }); renderShocks(); draw(); };
  $("shocks").oninput = e => { const k = e.target.closest(".shock")?.dataset.k, f = e.target.dataset.f; if (k === undefined || !f) return;
    S.shocks[k][f] = f === "kind" ? e.target.value : +e.target.value; draw(); };
  $("shocks").onclick = e => { if (e.target.dataset.del === undefined) return; S.shocks.splice(+e.target.dataset.del, 1); renderShocks(); draw(); };
  const cv = $("map");
  cv.onmousedown = e => { const c = cellAt(e); S.drag = [c, c]; };
  cv.onmousemove = e => {
    const [i, j] = cellAt(e), f = field();
    $("hover").textContent = `lat ${6.5 + i}, lon ${65.5 + j}` + (f ? `: ${f[i][j].toFixed(2)}` : "");
    if (S.drag) { S.drag[1] = [i, j]; draw(); }
  };
  cv.onmouseup = () => { if (!S.drag) return; const [lat0, lat1, lon0, lon1] = dragBox(); S.drag = null;
    S.shocks.push({ ch: 0, kind: "add", value: 3, lat0, lat1, lon0, lon1 }); renderShocks(); draw(); };
  await connect(); draw();
}
init();
