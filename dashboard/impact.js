// Impact module (starter). Every number here is computed from the LIVE rollout returned by the server.
// rollout: [steps][6][32][36] physical units; meta.land_sea_mask: [32][36] 0..1.
// Add new indicators by pushing to INDICATORS. Ideas: crop stress (rain deficit), flood risk (3-day rain),
// solar/wind generation, heat index (t2m + humidity proxy from tcwv), ensemble spread.
const IMPACT = (() => {
  const CH = { t2m: 0, tp: 1, u10: 2, v10: 3, msl: 4, tcwv: 5 };

  function landCells(meta, fn) {
    let n = 0;
    for (let i = 0; i < 32; i++) for (let j = 0; j < 36; j++) if (meta.land_sea_mask[i][j] > 0.5 && fn(i, j)) n++;
    return n;
  }

  const INDICATORS = [
    { name: "Heat stress cell-days (t2m > 40 C, land)", unit: "",
      calc: (r, meta) => r.reduce((s, day) => s + landCells(meta, (i, j) => day[CH.t2m][i][j] > 313.15), 0) },
    { name: "Heavy rain cell-days (> 64.5 mm/day, land)", unit: "",
      calc: (r, meta) => r.reduce((s, day) => s + landCells(meta, (i, j) => day[CH.tp][i][j] > 64.5), 0) },
    { name: "Mean land rain over the run", unit: "mm/day",
      calc: (r, meta) => {
        let s = 0, n = 0;
        for (const day of r) for (let i = 0; i < 32; i++) for (let j = 0; j < 36; j++)
          if (meta.land_sea_mask[i][j] > 0.5) { s += day[CH.tp][i][j]; n++; }
        return n ? s / n : 0;
      } },
    { name: "Mean 10 m wind speed (wind power proxy)", unit: "m/s",
      calc: (r) => {
        let s = 0, n = 0;
        for (const day of r) for (let i = 0; i < 32; i++) for (let j = 0; j < 36; j++) {
          s += Math.hypot(day[CH.u10][i][j], day[CH.v10][i][j]); n++;
        }
        return s / n;
      } },
  ];

  function render(el, base, shocked, meta) {
    if (!base) { el.innerHTML = '<p class="hint">Run a simulation to see impacts.</p>'; return; }
    const rows = INDICATORS.map(ind => {
      const b = ind.calc(base, meta), s = shocked ? ind.calc(shocked, meta) : null;
      const f = v => (v === null ? "-" : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)));
      return `<tr><td>${ind.name}${ind.unit ? " (" + ind.unit + ")" : ""}</td><td>${f(b)}</td><td>${f(s)}</td>
              <td>${s === null ? "-" : f(s - b)}</td></tr>`;
    });
    el.innerHTML = `<table><tr><th>Indicator</th><th>Prime</th><th>Shocked</th><th>Change</th></tr>${rows.join("")}</table>`;
  }

  return { render, INDICATORS };
})();
