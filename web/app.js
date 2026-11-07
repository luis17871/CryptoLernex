"use strict";
const $ = (id) => document.getElementById(id);
const C = { teal: "#2ee6c8", red: "#ff4d6d", gold: "#ffc24a", white: "#eef5fb", mute: "#5f7385", grid: "#0f1a24" };
const S = { state: null, replay: null, symbol: null, interval: null, revealId: null, revealT0: 0, hoverX: null };

const fmt = (v, d) => {
  if (v == null || isNaN(v)) return "-";
  const a = Math.abs(v); d = d ?? (a >= 1000 ? 0 : a >= 10 ? 2 : 4);
  return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
};
const pct = (v, d = 1) => (v == null || isNaN(v) ? "-" : (v * 100).toFixed(d) + "%");
const sgn = (v, d = 2) => (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%";
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()) || r.status);
  return r.json();
}

// ------------------------------------------------------------------ boot
async function boot() {
  const info = await api("/api/info");
  const st = info.settings;
  S.symbol = st.symbol; S.interval = st.interval;
  $("symbol").innerHTML = [...new Set([st.symbol, ...st.symbols])].map((s) => `<option>${esc(s)}</option>`).join("");
  $("interval").innerHTML = info.intervals.map((i) => `<option ${i === st.interval ? "selected" : ""}>${i}</option>`).join("");
  $("inH").value = st.horizon; $("inN").value = st.samples; $("inT").value = st.temperature; $("inP").value = st.top_p;
  $("symbol").onchange = (e) => { S.symbol = e.target.value; S.replay = null; refresh(); };
  $("interval").onchange = (e) => { S.interval = e.target.value; S.replay = null; refresh(); };
  const hz = [...new Set((st.tracks || []).map((t) => +t.split(":")[0]))];
  S.horizon = hz[0] || null;
  $("track").innerHTML = hz.map((h) => `<option value="${h}">horizonte ${h}</option>`).join("");
  $("track").hidden = hz.length < 2;
  $("track").onchange = (e) => { S.horizon = +e.target.value; $("inH").value = S.horizon; S.replay = null; refresh(); };
  if (S.horizon) $("inH").value = S.horizon;
  $("runBtn").onclick = run;
  setupEvents();
  setupHover();
  window.addEventListener("resize", draw);
  await refresh();
  setInterval(refresh, 10000);
  requestAnimationFrame(loop);
}

async function refresh() {
  try {
    let n = 160;
    if (S.replay) n = Math.min(1000, Math.ceil((Date.now() / 1000 - S.replay.anchor_ts) / S.replaySec) + 140);
    const st = await api(`/api/state?symbol=${S.symbol}&interval=${S.interval}&candles=${n}${S.horizon ? `&horizon=${S.horizon}` : ""}`);
    S.state = st;
    $("liveTxt").textContent = "live";
    render();
  } catch (e) {
    $("liveTxt").textContent = "offline";
    console.error(e);
  }
}

async function run() {
  const b = $("runBtn"); b.disabled = true; b.textContent = "SAMPLING...";
  try {
    S.replay = null;
    await api("/api/forecast", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: S.symbol, interval: S.interval, horizon: +$("inH").value,
        samples: +$("inN").value, temperature: +$("inT").value, top_p: +$("inP").value }),
    });
    await refresh();
  } catch (e) { alert("forecast failed: " + e.message); }
  b.disabled = false; b.textContent = "RUN";
}

// ------------------------------------------------------------------ panels
function currentRound() { return S.replay || (S.state && S.state.round); }

function render() {
  const st = S.state; if (!st) return;
  const info = st.info, eng = info.engine || {};
  const r = currentRound();
  $("source").textContent = `${info.source || "?"} - ${st.symbol} - ${st.interval}`;
  $("lastPx").textContent = st.candles.length ? fmt(st.candles[st.candles.length - 1][4]) : "-";
  $("engineName").textContent = `${eng.model || ""} - ${eng.device || ""}`;
  $("warn").textContent = info.warning || "";
  $("exportLnk").href = `/api/export.csv?symbol=${st.symbol}&interval=${st.interval}`;
  const isK = eng.engine === "kronos";
  $("modelCard").innerHTML = `
    <div class="h"><b>${esc(eng.model || "-")}</b><span class="${isK ? "ok" : "fb"}">${isK ? "loaded" : "fallback"}</span></div>
    <div class="row"><span>repo</span>${esc(eng.repo || "-")}</div>
    <div class="row"><span>params</span>${esc(eng.params || "-")}</div>
    <div class="row"><span>context</span>${r ? r.params.context : "-"} / ${eng.max_context || "-"}</div>
    <div class="row"><span>horizon</span>${r ? r.horizon : "-"}</div>
    <div class="row"><span>samples</span>${r ? r.params.samples : "-"}</div>
    <div class="row"><span>T / top_p</span>${r ? r.params.temperature + " / " + r.params.top_p : "-"}</div>
    <div class="row"><span>inference</span>${r ? r.params.seconds + "s" : "-"}</div>`;

  if (r && r.id !== S.revealId) { S.revealId = r.id; S.revealT0 = performance.now(); }
  if (r) {
    const s = r.summary, f = s.final;
    document.querySelectorAll(".H").forEach((e) => (e.textContent = r.horizon));
    $("roundTag").textContent = `ROUND #${r.id} ${S.replay ? "(replay)" : ""} / ${r.status === "done" ? "graded" : "reading the spread"}`;
    $("median").textContent = fmt(f.p50);
    $("medRet").textContent = sgn(s.median_return);
    $("medRet").style.color = s.median_return >= 0 ? C.teal : C.red;
    $("p90").textContent = fmt(f.p90); $("p10").textContent = fmt(f.p10);
    $("spread").textContent = `${fmt(s.spread)} (${pct(s.spread / s.last_close, 2)})`;
    $("var95").textContent = sgn(s.var95);
    $("nUp").textContent = s.n_up; $("nDn").textContent = s.n_down; $("nPaths").textContent = `${s.n_paths} futures`;
    const cells = 100, up = Math.round((s.n_up / s.n_paths) * cells);
    $("pathBar").innerHTML = Array.from({ length: cells }, (_, i) => `<i class="${i < up ? "u" : ""}"></i>`).join("");
    $("probUp").textContent = pct(s.prob_up, 0);
    const rg = $("regime"); rg.textContent = s.regime;
    rg.className = "regime " + (s.regime === "TRENDING UP" ? "up" : s.regime === "TRENDING DOWN" ? "dn" : "");
    $("trendVol").textContent = `${s.trend_score.toFixed(2)} / ${s.vol_ratio.toFixed(2)}x`;
    $("peakTrough").textContent = `step ${s.median_peak_step} / ${s.median_trough_step}`;
    const lv = [5, 3, 2, 1, 0.5];
    $("touch").innerHTML = "<tr><th>move</th><th>up</th><th>down</th></tr>" + lv.map((p) => {
      const u = s.touch[`+${p}%`], d = s.touch[`-${p}%`];
      return `<tr><td>±${p}%</td><td><span class="bar" style="width:${u * 50}px;background:${C.teal}"></span>${pct(u, 0)}</td>
        <td><span class="bar" style="width:${d * 50}px;background:${C.red}"></span>${pct(d, 0)}</td></tr>`;
    }).join("");
  }
  renderScore(st);
  renderRounds(st);
  draw();
}

function renderScore(st) {
  const m = st.metrics, k = m.primary, b = m.shadow;
  $("openRounds").textContent = `${k.rounds || 0} graded - ${k.open || 0} live`;
  const name = (st.info.engine || {}).model || "model";
  const rows = [
    ["hit-rate (dir)", "hit_rate", pct, 1], ["p10-p90 cover", "coverage", pct, 0], ["CRPS skill", "crps_skill", pct, 1],
    ["MAE skill", "skill", pct, 1],
  ];
  const best = (key, sign) => {
    if (!k.rounds || !b.rounds) return "";
    if (key === "coverage") return Math.abs(k[key] - 0.8) <= Math.abs(b[key] - 0.8) ? "k" : "b";
    return k[key] * sign >= b[key] * sign ? "k" : "b";
  };
  $("scoreTbl").innerHTML = `<tr><th></th><th>${esc(name)}</th><th>baseline</th></tr>` + rows.map(([l, key, f, sgn_]) => {
    const w = best(key, sgn_);
    return `<tr><td>${l}</td><td class="${w === "k" ? "best" : ""}">${k.rounds ? f(k[key]) : "-"}</td>
      <td class="${w === "b" ? "best" : ""}">${b.rounds ? f(b[key]) : "-"}</td></tr>`;
  }).join("") + sigRows(k, b) + `<tr><td>target</td><td colspan=2 style="text-align:center;color:#3f5466">hit &gt;50% - cover ≈80% - skill &gt;0</td></tr>`;
  drawPit(k.pit_hist);
}

// Significance on non-overlapping rounds: the honest sample size
function sigRows(k, b) {
  const ci = (m) => (m.n_indep ? `${pct(m.hit_indep, 0)} <span style="color:#5f7385">[${pct(m.ci_low, 0)}-${pct(m.ci_high, 0)}]</span>` : "-");
  const cls = { "ventaja": "hit", "peor que azar": "miss", "no concluyente": "open", "pocos datos": "open" };
  const v = (m) => (m.verdict ? `<span class="pill ${cls[m.verdict] || "open"}">${esc(m.verdict)}</span>` : "-");
  return `<tr><td>indep. rounds</td><td>${k.n_indep ?? 0}</td><td>${b.n_indep ?? 0}</td></tr>
    <tr><td>hit indep. (95% CI)</td><td>${ci(k)}</td><td>${ci(b)}</td></tr>
    <tr><td>p-value vs coin</td><td>${k.n_indep ? k.p_value.toFixed(3) : "-"}</td><td>${b.n_indep ? b.p_value.toFixed(3) : "-"}</td></tr>
    <tr><td>verdict</td><td>${v(k)}</td><td>${v(b)}</td></tr>`;
}

function renderRounds(st) {
  const sel = currentRound();
  $("roundsTbl").innerHTML = `<tr><th>#</th><th>anchor (UTC)</th><th>last</th><th>median</th><th>P(up)</th><th>regime</th>
    <th>real</th><th>cover</th><th>dir</th></tr>` + st.rounds.map((r) => {
    const s = r.summary, sc = r.score;
    const dir = !sc ? '<span class="pill open">live</span>' :
      `<span class="pill ${sc.direction_hit ? "hit" : "miss"}">${sc.direction_hit ? "HIT" : "MISS"}</span>${r.status === "open" ? "*" : ""}`;
    const t = new Date(r.anchor_ts * 1000).toISOString().slice(5, 16).replace("T", " ");
    return `<tr data-id="${r.id}" class="${sel && sel.id === r.id ? "sel" : ""}"><td>${r.id}</td><td>${t}</td>
      <td>${fmt(r.last_close)}</td><td>${fmt(s.median)}</td><td>${pct(s.prob_up, 0)}</td><td>${s.regime}</td>
      <td>${sc ? fmt(sc.actual_last) : "-"}</td><td>${sc ? pct(sc.coverage, 0) : "-"}</td><td>${dir}</td></tr>`;
  }).join("");
  $("roundsTbl").querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = async () => {
    const id = +tr.dataset.id;
    if (S.state.round && id === S.state.round.id) { S.replay = null; return refresh(); }
    S.replay = await api(`/api/rounds/${id}`); S.replaySec = S.state.seconds;
    refresh();
  }));
}

function drawPit(hist) {
  const cv = $("pit"), dpr = devicePixelRatio || 1, w = cv.clientWidth, h = 60;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  if (!hist) { g.fillStyle = C.mute; g.fillText("no graded rounds yet", 4, 34); return; }
  const bw = w / hist.length, mx = Math.max(0.2, ...hist);
  hist.forEach((v, i) => {
    const bh = (v / mx) * (h - 10);
    g.fillStyle = "#2ee6c855"; g.fillRect(i * bw + 1, h - bh, bw - 2, bh);
  });
  const y = h - (0.1 / mx) * (h - 10);
  g.strokeStyle = C.gold; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
}

// ------------------------------------------------------------------ chart
function geometry() {
  const cv = $("chart"), st = S.state, r = currentRound();
  const W = cv.clientWidth, H = cv.clientHeight;
  const pad = { l: 12, r: 72, t: 40, b: 26 };
  const sec = st.seconds;
  let anchorIdx = st.candles.length - 1;
  if (r) {
    anchorIdx = st.candles.findIndex((c) => c[0] === r.anchor_ts);
    if (anchorIdx < 0) anchorIdx = st.candles.length - 1;
  }
  const Hh = r ? r.horizon : 0;
  const nHist = Math.max(40, Math.min(anchorIdx + 1, Math.round(Math.max(Hh, 24) * 2.2)));
  const hist = st.candles.slice(anchorIdx + 1 - nHist, anchorIdx + 1);
  const real = r ? st.candles.slice(anchorIdx + 1, anchorIdx + 1 + Hh) : [];
  const slots = nHist + Hh + 1;
  const bw = (W - pad.l - pad.r) / slots;
  const x = (i) => pad.l + (i + 0.5) * bw; // i: 0..nHist-1 history, nHist-1 = anchor
  let lo = Infinity, hi = -Infinity;
  hist.forEach((c) => { lo = Math.min(lo, c[3]); hi = Math.max(hi, c[2]); });
  real.forEach((c) => { lo = Math.min(lo, c[3]); hi = Math.max(hi, c[2]); });
  if (r) {
    const b = r.summary.band;
    lo = Math.min(lo, ...b.p5); hi = Math.max(hi, ...b.p95);
  }
  const m = (hi - lo) * 0.08; lo -= m; hi += m;
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const yInv = (py) => lo + (1 - (py - pad.t) / (H - pad.t - pad.b)) * (hi - lo);
  return { cv, W, H, pad, bw, x, y, yInv, hist, real, nHist, r, sec, lo, hi };
}

function loop() { if (S.state && currentRound() && performance.now() - S.revealT0 < 1600) draw(); requestAnimationFrame(loop); }

function draw() {
  const st = S.state; if (!st || !st.candles.length) return;
  const G = geometry(), { cv, W, H, pad, bw, x, y, hist, real, nHist, r } = G;
  const dpr = devicePixelRatio || 1;
  if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  // background
  const bg = g.createLinearGradient(0, 0, 0, H); bg.addColorStop(0, "#071019"); bg.addColorStop(1, "#04070b");
  g.fillStyle = bg; g.fillRect(0, 0, W, H);
  // grid + axis
  g.font = "10px JetBrains Mono, monospace"; g.textBaseline = "middle";
  const ticks = niceTicks(G.lo, G.hi, 7);
  ticks.forEach((v) => {
    const yy = y(v); g.strokeStyle = C.grid; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad.l, yy); g.lineTo(W - pad.r, yy); g.stroke();
    g.fillStyle = C.mute; g.fillText(fmt(v), W - pad.r + 8, yy);
  });
  for (let i = 0; i < nHist + (r ? r.horizon : 0); i += Math.max(1, Math.round(nHist / 6))) {
    g.strokeStyle = "#0b141c"; g.beginPath(); g.moveTo(x(i), pad.t); g.lineTo(x(i), H - pad.b); g.stroke();
  }
  // history candles
  hist.forEach((c, i) => candle(g, x(i), bw, y, c, c[4] >= c[1] ? C.teal : C.red, 0.95));

  const ax = x(nHist - 1);
  if (r) {
    const s = r.summary, Hh = r.horizon, last = r.last_close;
    const prog = Math.min(1, (performance.now() - S.revealT0) / 1400);
    const kmax = Math.max(1, Math.ceil(prog * Hh));
    const fx = (k) => x(nHist - 1 + k); // k = 0 anchor, 1..H forecast
    // bands
    const band = (a, b, col) => {
      g.beginPath(); g.moveTo(fx(0), y(last));
      for (let k = 1; k <= kmax; k++) g.lineTo(fx(k), y(s.band[a][k - 1]));
      for (let k = kmax; k >= 1; k--) g.lineTo(fx(k), y(s.band[b][k - 1]));
      g.closePath(); g.fillStyle = col; g.fill();
    };
    band("p90", "p10", "rgba(46,230,200,0.07)");
    band("p75", "p25", "rgba(46,230,200,0.08)");
    // paths
    g.globalCompositeOperation = "lighter"; g.lineWidth = 1;
    const alpha = Math.max(0.07, Math.min(0.4, 14 / r.paths.length));
    r.paths.forEach((p) => {
      const up = p[Hh - 1] > last;
      g.strokeStyle = up ? `rgba(46,230,200,${alpha})` : `rgba(255,77,109,${alpha})`;
      g.beginPath(); g.moveTo(fx(0), y(last));
      for (let k = 1; k <= kmax; k++) g.lineTo(fx(k), y(p[k - 1]));
      g.stroke();
    });
    // origin glow
    const rg = g.createRadialGradient(ax, y(last), 0, ax, y(last), 40);
    rg.addColorStop(0, "rgba(140,255,240,0.55)"); rg.addColorStop(1, "rgba(46,230,200,0)");
    g.fillStyle = rg; g.fillRect(ax - 40, y(last) - 40, 80, 80);
    g.globalCompositeOperation = "source-over";
    // p10 / p90 lines
    [["p90", C.teal], ["p10", C.red]].forEach(([q, col]) => {
      g.strokeStyle = col; g.globalAlpha = 0.55; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(fx(0), y(last));
      for (let k = 1; k <= kmax; k++) g.lineTo(fx(k), y(s.band[q][k - 1]));
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    });
    // median
    g.save(); g.shadowColor = C.gold; g.shadowBlur = 14; g.strokeStyle = C.gold; g.lineWidth = 2.2;
    g.beginPath(); g.moveTo(fx(0), y(last));
    for (let k = 1; k <= kmax; k++) g.lineTo(fx(k), y(s.band.p50[k - 1]));
    g.stroke(); g.restore();
    // realised candles (the truth arriving)
    real.forEach((c, i) => {
      const k = i + 1, inside = c[4] >= s.band.p10[i] && c[4] <= s.band.p90[i];
      candle(g, fx(k), bw, y, c, C.white, 0.9);
      if (!inside) { g.strokeStyle = C.gold; g.beginPath(); g.arc(fx(k), y(c[4]), 4, 0, 7); g.stroke(); }
    });
    // anchor line
    g.strokeStyle = "rgba(46,230,200,0.5)"; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(ax, pad.t - 10); g.lineTo(ax, H - pad.b); g.stroke(); g.setLineDash([]);
    tag(g, ax, H - pad.b + 12, "NOW", "#0c2526", C.teal, "center");
    // price tags on the right
    const tags = [[s.final.p90, C.teal, "p90"], [s.final.p50, C.gold, "median"], [s.final.p10, C.red, "p10"]];
    tags.forEach(([v, col, l]) => {
      const yy = y(v);
      tag(g, fx(Hh) - 4, yy, `${l} ${fmt(v)}`, "rgba(5,10,16,0.9)", col, "right");
      tag(g, W - pad.r + 2, yy, fmt(v), col, "#05080d", "left");
    });
  }
  // last price tag
  const lc = st.candles[st.candles.length - 1];
  tag(g, W - pad.r + 2, y(lc[4]), fmt(lc[4]), "#1f3140", C.white, "left");
  // time labels
  g.fillStyle = C.mute; g.textBaseline = "alphabetic";
  const step = Math.max(1, Math.ceil(nHist / Math.max(2, Math.floor((W - pad.r) / 190))));
  for (let i = 0; i < nHist - step / 2; i += step) g.fillText(tlabel(hist[i][0], G.sec), x(i) - 16, H - 6);
  if (r) g.fillText(tlabel(r.anchor_ts + r.horizon * G.sec, G.sec), x(nHist - 1 + r.horizon) - 40, H - 6);
  drawHover(g, G);
}

function candle(g, cx, bw, y, c, col, a) {
  const w = Math.max(1, bw * 0.62);
  g.globalAlpha = a; g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 1;
  g.beginPath(); g.moveTo(cx, y(c[2])); g.lineTo(cx, y(c[3])); g.stroke();
  const t = y(Math.max(c[1], c[4])), b = y(Math.min(c[1], c[4]));
  g.fillRect(cx - w / 2, t, w, Math.max(1, b - t));
  g.globalAlpha = 1;
}

function tag(g, px, py, txt, bg, fg, align) {
  g.font = "10px JetBrains Mono, monospace"; g.textBaseline = "middle";
  const w = g.measureText(txt).width + 10, h = 16;
  const x0 = align === "right" ? px - w : align === "center" ? px - w / 2 : px;
  g.fillStyle = bg; g.fillRect(x0, py - h / 2, w, h);
  g.strokeStyle = fg; g.globalAlpha = 0.6; g.strokeRect(x0 + 0.5, py - h / 2 + 0.5, w - 1, h - 1); g.globalAlpha = 1;
  g.fillStyle = fg; g.fillText(txt, x0 + 5, py);
}

function tlabel(ts, sec) {
  const d = new Date(ts * 1000).toISOString();
  return sec >= 86400 ? d.slice(5, 10) : d.slice(5, 10).replace("-", "/") + " " + d.slice(11, 16);
}

function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
  return out;
}

// ------------------------------------------------------------------ hover
function setupHover() {
  const cv = $("chart");
  cv.addEventListener("mousemove", (e) => { const b = cv.getBoundingClientRect(); S.hoverX = e.clientX - b.left; S.hoverY = e.clientY - b.top; draw(); });
  cv.addEventListener("mouseleave", () => { S.hoverX = null; $("tip").style.display = "none"; draw(); });
}

function drawHover(g, G) {
  const tip = $("tip");
  if (S.hoverX == null) return;
  const { x, y, yInv, bw, pad, nHist, hist, r, H, W } = G;
  const i = Math.round((S.hoverX - pad.l) / bw - 0.5);
  if (i < 0 || S.hoverX > W - pad.r) { tip.style.display = "none"; return; }
  g.strokeStyle = "#2a3e50"; g.beginPath(); g.moveTo(x(i), pad.t); g.lineTo(x(i), H - pad.b); g.stroke();
  g.beginPath(); g.moveTo(pad.l, S.hoverY); g.lineTo(W - pad.r, S.hoverY); g.stroke();
  const price = yInv(S.hoverY);
  let html = `<b>${fmt(price)}</b><br>`;
  if (i < nHist && hist[i]) {
    const c = hist[i];
    html += `${tlabel(c[0], G.sec)}<br>O ${fmt(c[1])} H ${fmt(c[2])}<br>L ${fmt(c[3])} C ${fmt(c[4])}`;
  } else if (r) {
    const k = i - (nHist - 1);
    if (k >= 1 && k <= r.horizon) {
      const s = r.summary, col = r.paths.map((p) => p[k - 1]);
      const above = col.filter((v) => v >= price).length / col.length;
      html += `step +${k} - ${tlabel(r.anchor_ts + k * G.sec, G.sec)}<br>p90 ${fmt(s.band.p90[k - 1])}<br>
        <span style="color:${C.gold}">p50 ${fmt(s.band.p50[k - 1])}</span><br>p10 ${fmt(s.band.p10[k - 1])}<br>
        P(close &ge; cursor) <b>${pct(above, 0)}</b>`;
      const rc = G.real[k - 1]; if (rc) html += `<br>real <b>${fmt(rc[4])}</b>`;
    } else { tip.style.display = "none"; return; }
  }
  tip.innerHTML = html; tip.style.display = "block";
  const tx = S.hoverX + 14 + 180 > W ? S.hoverX - 190 : S.hoverX + 14;
  tip.style.left = tx + "px"; tip.style.top = Math.max(40, S.hoverY - 40) + "px";
}

// ------------------------------------------------------------------ log stream
function setupEvents() {
  const el = $("log");
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    const d = document.createElement("div");
    d.className = ev.level;
    const t = new Date(ev.t * 1000).toTimeString().slice(0, 8);
    d.innerHTML = `<span class="t">${t}</span><span class="g">${esc(ev.tag)}</span><span class="m">${esc(ev.msg)}</span>`;
    const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    el.appendChild(d);
    while (el.children.length > 300) el.removeChild(el.firstChild);
    if (stick) el.scrollTop = el.scrollHeight;
    if (ev.tag === "score" || /paths up/.test(ev.msg)) setTimeout(refresh, 300);
  };
}

boot();
