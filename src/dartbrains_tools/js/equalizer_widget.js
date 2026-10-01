// Equalizer Widget - Web Audio + Canvas 2D
// Looping audio player with a graphic equalizer drawn as a filter's frequency response.
//
// Each band is a BiquadFilterNode (low shelf, peaking..., high shelf) in series.
// A band's gain is linear in [0, 1] (1 = pass, 0 = remove) and is converted to a dB
// cut for the filter. The curve under the handles is the *actual* combined response
// of the chain (product of each node's getFrequencyResponse), plotted as linear gain
// against log frequency -- the same kind of plot as scipy's freqz in the chapter.

const W = 760;
const WAVE_H = 96;
const EQ_H = 260;
const MIN_DB = -40; // gain 0 -> -40 dB (1% amplitude); a biquad cannot reach exactly 0
const F_LO = 20;
const F_HI = 20000;

const C = {
  ink: "#111827",
  muted: "#6b7280",
  faint: "#9ca3af",
  line: "#e5e7eb",
  track: "#eef0f3",
  accent: "#4f6ef7",
  accentDark: "#25307a",
  accentSoft: "rgba(79, 110, 247, 0.10)",
  spectrum: "rgba(17, 24, 39, 0.07)",
};

const PRESETS = {
  Flat: (n) => Array(n).fill(1),
  "Low-pass": (n) => Array.from({ length: n }, (_, i) => (i < n / 2 ? 1 : 0)),
  "High-pass": (n) => Array.from({ length: n }, (_, i) => (i < n / 2 ? 0 : 1)),
  "Band-pass": (n) =>
    Array.from({ length: n }, (_, i) => (Math.abs(i - (n - 1) / 2) < 1.5 ? 1 : 0)),
  "Band-stop": (n) =>
    Array.from({ length: n }, (_, i) => (Math.abs(i - (n - 1) / 2) < 1.5 ? 0 : 1)),
};

const ICONS = {
  upload:
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
  play:
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><polygon points="6 4 20 12 6 20 6 4"/></svg>',
  pause:
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="8" y1="5" x2="8" y2="19"/><line x1="16" y1="5" x2="16" y2="19"/></svg>',
};

const CSS = `
.dbeq { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  color: ${C.ink}; max-width: ${W + 42}px; box-sizing: border-box; padding: 20px;
  border: 1px solid ${C.line}; border-radius: 14px; background: #fff;
  transition: border-color .15s, background .15s; }
.dbeq.dbeq-over { border-color: ${C.accent}; background: #f7f8ff; }
.dbeq * { box-sizing: border-box; }
.dbeq-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.dbeq-btn { display: inline-flex; align-items: center; gap: 8px; border: none;
  background: #f3f4f6; color: ${C.ink}; font: 500 14px/1 inherit; font-family: inherit;
  padding: 9px 14px; border-radius: 8px; cursor: pointer; }
.dbeq-btn:hover { background: #e9eaee; }
.dbeq-btn:disabled { opacity: .45; cursor: default; }
.dbeq-name { font-size: 14px; color: ${C.ink}; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; max-width: 420px; }
.dbeq-name.dbeq-empty { color: ${C.faint}; }
.dbeq-time { font-size: 14px; color: ${C.muted}; font-variant-numeric: tabular-nums; }
.dbeq-wave { position: relative; margin: 16px 0 14px; }
.dbeq-drop { position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; border: 1.5px dashed #d1d5db; border-radius: 10px;
  color: ${C.faint}; font-size: 14px; pointer-events: none; }
.dbeq canvas { display: block; max-width: 100%; height: auto; }
.dbeq-wave canvas { cursor: pointer; }
.dbeq-eq { margin-top: 18px; }
.dbeq-eq canvas { touch-action: none; outline: none; }
.dbeq-head { display: flex; align-items: center; justify-content: space-between;
  gap: 12px; flex-wrap: wrap; margin-top: 18px; }
.dbeq-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.dbeq-chip { border: 1px solid ${C.line}; background: #fff; color: ${C.muted};
  font: 500 12px/1 inherit; font-family: inherit; padding: 6px 10px; border-radius: 999px;
  cursor: pointer; }
.dbeq-chip:hover { color: ${C.ink}; border-color: #d1d5db; }
.dbeq-chip.dbeq-on { color: ${C.accent}; border-color: ${C.accent}; background: #f5f7ff; }
.dbeq-legend { font-size: 12px; color: ${C.muted}; display: flex; gap: 14px; }
.dbeq-legend i { display: inline-block; width: 14px; height: 2px; vertical-align: middle;
  margin-right: 6px; background: ${C.accentDark}; }
.dbeq-legend i.dbeq-spec { height: 8px; background: rgba(17,24,39,.12); border-radius: 2px; }
`;

function makeCanvas(w, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = document.createElement("canvas");
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { canvas, ctx };
}

function fmtTime(s) {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

function fmtHz(f) {
  return f >= 1000 ? `${+(f / 1000).toFixed(1)}k` : `${Math.round(f)}`;
}

function el_(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

export default {
  render({ model, el }) {
    const style = el_("style");
    style.textContent = CSS;
    el.appendChild(style);

    const root = el_("div", "dbeq");
    el.appendChild(root);

    // --- top row: browse + filename --------------------------------------------
    const top = el_("div", "dbeq-row");
    const browse = el_("button", "dbeq-btn", `${ICONS.upload}<span>Browse</span>`);
    const name = el_("span", "dbeq-name dbeq-empty", "No file loaded");
    const input = el_("input");
    input.type = "file";
    input.accept = "audio/*";
    input.style.display = "none";
    top.append(browse, name, input);
    root.appendChild(top);

    // --- waveform ---------------------------------------------------------------
    const waveWrap = el_("div", "dbeq-wave");
    const wave = makeCanvas(W, WAVE_H);
    const drop = el_("div", "dbeq-drop", "Drop an audio file here");
    waveWrap.append(wave.canvas, drop);
    root.appendChild(waveWrap);

    // --- transport --------------------------------------------------------------
    const transport = el_("div", "dbeq-row");
    const playBtn = el_("button", "dbeq-btn", `${ICONS.play}<span>Play</span>`);
    playBtn.disabled = true;
    const time = el_("span", "dbeq-time", "0:00 / 0:00");
    transport.append(playBtn, time);
    root.appendChild(transport);

    // --- presets + legend -------------------------------------------------------
    const head = el_("div", "dbeq-head");
    const chips = el_("div", "dbeq-chips");
    const chipEls = {};
    for (const label of Object.keys(PRESETS)) {
      const chip = el_("button", "dbeq-chip", label);
      chip.addEventListener("click", () => {
        setGains(PRESETS[label](bands.length));
        commit();
      });
      chipEls[label] = chip;
      chips.appendChild(chip);
    }
    const legend = el_(
      "div",
      "dbeq-legend",
      '<span><i></i>Filter response</span><span><i class="dbeq-spec"></i>Output spectrum</span>'
    );
    head.append(chips, legend);
    root.appendChild(head);

    // --- equalizer --------------------------------------------------------------
    const eqWrap = el_("div", "dbeq-eq");
    const eq = makeCanvas(W, EQ_H);
    eq.canvas.tabIndex = 0;
    eq.canvas.setAttribute("aria-label", "Equalizer: arrow keys select a band and change its gain");
    eqWrap.appendChild(eq.canvas);
    root.appendChild(eqWrap);

    // --- state ------------------------------------------------------------------
    let bands = (model.get("bands") || []).slice();
    let gains = normGains(model.get("gains"));
    let audioCtx = null;
    let chain = []; // live BiquadFilterNodes
    let analyser = null;
    let buffer = null;
    let peaks = null;
    let source = null;
    let playing = false;
    let startedAt = 0;
    let offset = 0;
    let raf = 0;
    let dragBand = -1;
    let focusBand = -1;
    let spectrum = null;

    // Filters on an OfflineAudioContext give the response curve before any audio exists.
    const respCtx = new OfflineAudioContext(1, 128, 48000);
    let respChain = [];
    const NF = 360;
    const freqs = new Float32Array(NF);
    for (let i = 0; i < NF; i++) freqs[i] = F_LO * Math.pow(F_HI / F_LO, i / (NF - 1));
    const mag = new Float32Array(NF);
    const phase = new Float32Array(NF);
    let response = new Float32Array(NF).fill(1);

    function normGains(g) {
      const n = bands.length;
      const out = Array.from({ length: n }, (_, i) => (g && g[i] != null ? +g[i] : 1));
      return out.map((v) => Math.min(1, Math.max(0, isFinite(v) ? v : 1)));
    }

    // Octave-spaced bands -> Q ~ 1.41. Derived from spacing so custom `bands` also work.
    function bandQ() {
      if (bands.length < 2) return 1.41;
      const ratio = Math.pow(bands[bands.length - 1] / bands[0], 1 / (bands.length - 1));
      const octaves = Math.log2(ratio);
      return Math.sqrt(Math.pow(2, octaves)) / (Math.pow(2, octaves) - 1);
    }

    function configure(node, i) {
      const n = bands.length;
      node.type = i === 0 ? "lowshelf" : i === n - 1 ? "highshelf" : "peaking";
      node.frequency.value = bands[i];
      node.Q.value = bandQ();
      node.gain.value = filterDb[i] ?? gainToDb(gains[i]);
    }

    function gainToDb(g) {
      return Math.max(MIN_DB, 20 * Math.log10(Math.max(g, 1e-6)));
    }

    function buildChain(ctx) {
      return bands.map((_, i) => {
        const node = ctx.createBiquadFilter();
        configure(node, i);
        return node;
      });
    }

    function wireLive() {
      if (!audioCtx) return;
      for (const n of chain) n.disconnect();
      chain = buildChain(audioCtx);
      for (let i = 0; i < chain.length - 1; i++) chain[i].connect(chain[i + 1]);
      if (chain.length) chain[chain.length - 1].connect(analyser);
      if (source) {
        source.disconnect();
        source.connect(chain[0] || analyser);
      }
    }

    function rebuildBands() {
      bands = (model.get("bands") || []).slice();
      gains = normGains(gains);
      respChain = buildChain(respCtx);
      wireLive();
      updateResponse();
    }

    // Neighbouring bands overlap, so setting each filter to its own slider value makes
    // the cuts pile up (two passed bands between deep cuts barely rise above 0). Solve
    // for per-filter dB gains whose *combined* response hits every slider at its band
    // centre: measure the error there, correct, repeat (graphic-EQ gain matching).
    let filterDb = [];
    function solveFilterDb() {
      const n = bands.length;
      const target = gains.map(gainToDb);
      const centres = new Float32Array(bands);
      const m = new Float32Array(n);
      const p = new Float32Array(n);
      const db = target.slice();
      for (let it = 0; it < 30; it++) {
        const total = new Float64Array(n);
        respChain.forEach((node, i) => {
          node.gain.value = db[i];
          node.getFrequencyResponse(centres, m, p);
          for (let j = 0; j < n; j++) total[j] += 20 * Math.log10(Math.max(m[j], 1e-9));
        });
        let worst = 0;
        for (let i = 0; i < n; i++) {
          const err = target[i] - total[i];
          worst = Math.max(worst, Math.abs(err));
          db[i] = Math.min(24, Math.max(-60, db[i] + 0.8 * err));
        }
        if (worst < 0.1) break;
      }
      return db;
    }

    function updateResponse() {
      filterDb = solveFilterDb();
      response = new Float32Array(NF).fill(1);
      respChain.forEach((node, i) => {
        node.gain.value = filterDb[i];
        node.getFrequencyResponse(freqs, mag, phase);
        for (let k = 0; k < NF; k++) response[k] *= mag[k];
      });
      chain.forEach((node, i) => {
        node.gain.setTargetAtTime(filterDb[i], audioCtx.currentTime, 0.015);
      });
      for (const [label, fn] of Object.entries(PRESETS)) {
        const p = fn(bands.length);
        const on = p.every((v, i) => Math.abs(v - gains[i]) < 1e-3);
        chipEls[label].classList.toggle("dbeq-on", on);
      }
      drawEq();
    }

    function setGains(g) {
      gains = normGains(g);
      updateResponse();
    }

    function commit() {
      model.set("gains", gains.slice());
      model.save_changes();
    }

    // --- audio ------------------------------------------------------------------
    function ensureCtx() {
      if (audioCtx) return audioCtx;
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 8192;
      analyser.smoothingTimeConstant = 0.8;
      analyser.connect(audioCtx.destination);
      spectrum = new Float32Array(analyser.frequencyBinCount);
      wireLive();
      return audioCtx;
    }

    function position() {
      if (!buffer) return 0;
      if (!playing) return offset;
      return (audioCtx.currentTime - startedAt) % buffer.duration;
    }

    function start(at) {
      const ctx = ensureCtx();
      if (ctx.state === "suspended") ctx.resume();
      stopSource();
      source = ctx.createBufferSource();
      source.buffer = buffer;
      source.loop = true;
      source.connect(chain[0] || analyser);
      source.start(0, at);
      startedAt = ctx.currentTime - at;
      playing = true;
      playBtn.innerHTML = `${ICONS.pause}<span>Pause</span>`;
      loop();
    }

    function stopSource() {
      if (source) {
        try { source.stop(); } catch (e) { /* already stopped */ }
        source.disconnect();
        source = null;
      }
    }

    function pause() {
      offset = position();
      stopSource();
      playing = false;
      playBtn.innerHTML = `${ICONS.play}<span>Play</span>`;
      drawWave();
      drawEq();
    }

    async function loadFile(file) {
      if (!file) return;
      const ctx = ensureCtx();
      let decoded;
      try {
        decoded = await ctx.decodeAudioData(await file.arrayBuffer());
      } catch (e) {
        name.textContent = `Could not decode ${file.name}`;
        name.classList.add("dbeq-empty");
        return;
      }
      if (playing) pause();
      buffer = decoded;
      peaks = computePeaks(buffer, Math.floor(W / 3));
      offset = 0;
      name.textContent = file.name;
      name.classList.remove("dbeq-empty");
      drop.style.display = "none";
      playBtn.disabled = false;
      model.set("filename", file.name);
      model.save_changes();
      start(0);
    }

    function computePeaks(buf, n) {
      const chans = [];
      for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
      const len = buf.length;
      const out = new Float32Array(n);
      const step = len / n;
      let max = 1e-9;
      for (let i = 0; i < n; i++) {
        const a = Math.floor(i * step);
        const b = Math.min(len, Math.floor((i + 1) * step));
        let sum = 0;
        let cnt = 0;
        for (let j = a; j < b; j += 16) {
          let v = 0;
          for (const ch of chans) v += ch[j];
          v /= chans.length;
          sum += v * v;
          cnt++;
        }
        out[i] = Math.sqrt(sum / Math.max(cnt, 1)); // RMS reads better than raw peaks
        if (out[i] > max) max = out[i];
      }
      for (let i = 0; i < n; i++) out[i] /= max;
      return out;
    }

    // --- drawing ----------------------------------------------------------------
    function drawWave() {
      const ctx = wave.ctx;
      ctx.clearRect(0, 0, W, WAVE_H);
      if (!peaks) return;
      const mid = WAVE_H / 2;
      const frac = buffer ? position() / buffer.duration : 0;
      const n = peaks.length;
      const bw = W / n;
      for (let i = 0; i < n; i++) {
        const h = Math.max(1.5, peaks[i] * (WAVE_H - 6));
        ctx.fillStyle = i / n < frac ? C.accentDark : C.accent;
        ctx.fillRect(i * bw, mid - h / 2, Math.max(1, bw - 1), h);
      }
      ctx.fillStyle = C.ink;
      ctx.fillRect(Math.round(frac * W), 0, 1.5, WAVE_H);
      time.textContent = `${fmtTime(position())} / ${fmtTime(buffer.duration)}`;
    }

    // EQ geometry
    const PL = 44;
    const PR = W - 18;
    const PT = 14;
    const PB = EQ_H - 36;
    const fx = (f) => PL + (Math.log(f / F_LO) / Math.log(F_HI / F_LO)) * (PR - PL);
    const gy = (g) => PB - g * (PB - PT);
    const yg = (y) => Math.min(1, Math.max(0, (PB - y) / (PB - PT)));

    function drawEq() {
      const ctx = eq.ctx;
      ctx.clearRect(0, 0, W, EQ_H);
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";

      // y grid + labels
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const g of [0, 0.25, 0.5, 0.75, 1]) {
        ctx.strokeStyle = g === 0 ? "#d1d5db" : C.track;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(PL, Math.round(gy(g)) + 0.5);
        ctx.lineTo(PR, Math.round(gy(g)) + 0.5);
        ctx.stroke();
        if (g === 0 || g === 0.5 || g === 1) {
          ctx.fillStyle = C.faint;
          ctx.fillText(g.toFixed(g === 0.5 ? 1 : 0), PL - 10, gy(g));
        }
      }
      ctx.save();
      ctx.translate(11, (PT + PB) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillStyle = C.faint;
      ctx.fillText("Gain", 0, 0);
      ctx.restore();

      // live output spectrum (filtered audio), dB mapped to [0, 1]
      if (playing && analyser) {
        analyser.getFloatFrequencyData(spectrum);
        const binHz = audioCtx.sampleRate / analyser.fftSize;
        ctx.beginPath();
        ctx.moveTo(PL, PB);
        for (let k = 0; k < NF; k += 2) {
          const bin = Math.min(spectrum.length - 1, Math.round(freqs[k] / binHz));
          const v = Math.min(1, Math.max(0, (spectrum[bin] + 100) / 75));
          ctx.lineTo(fx(freqs[k]), gy(v));
        }
        ctx.lineTo(PR, PB);
        ctx.closePath();
        ctx.fillStyle = C.spectrum;
        ctx.fill();
      }

      // filter response
      ctx.beginPath();
      ctx.moveTo(fx(freqs[0]), gy(Math.min(1.05, response[0])));
      for (let k = 1; k < NF; k++) ctx.lineTo(fx(freqs[k]), gy(Math.min(1.05, response[k])));
      ctx.lineTo(PR, PB);
      ctx.lineTo(PL, PB);
      ctx.closePath();
      ctx.fillStyle = C.accentSoft;
      ctx.fill();
      ctx.beginPath();
      for (let k = 0; k < NF; k++) {
        const x = fx(freqs[k]);
        const y = gy(Math.min(1.05, response[k]));
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.strokeStyle = C.accentDark;
      ctx.lineWidth = 1.75;
      ctx.stroke();

      // band sliders: track, fill from 0 to handle, handle
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      bands.forEach((f, i) => {
        const x = Math.round(fx(f)) + 0.5;
        const y = gy(gains[i]);
        ctx.strokeStyle = "rgba(79, 110, 247, 0.55)";
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(x, PB);
        ctx.lineTo(x, y);
        ctx.stroke();
        const active = i === dragBand || i === focusBand;
        ctx.beginPath();
        ctx.arc(x, y, active ? 8 : 7, 0, Math.PI * 2);
        ctx.fillStyle = active ? C.accent : "#fff";
        ctx.fill();
        ctx.lineWidth = 1.75;
        ctx.strokeStyle = C.accent;
        ctx.stroke();
        ctx.fillStyle = active ? C.ink : C.muted;
        ctx.fillText(`${fmtHz(f)}`, x, PB + 10);
        if (active) {
          ctx.fillStyle = C.ink;
          ctx.textBaseline = "bottom";
          ctx.fillText(gains[i].toFixed(2), x, Math.max(PT + 2, y - 11));
          ctx.textBaseline = "top";
        }
      });
      ctx.fillStyle = C.faint;
      ctx.textAlign = "right";
      ctx.fillText("Frequency (Hz)", PR, PB + 24);
    }

    let _animateErrLogged = false;
    function loop() {
      cancelAnimationFrame(raf);
      const tick = () => {
        try {
          drawWave();
          drawEq();
        } catch (e) {
          if (!_animateErrLogged) {
            _animateErrLogged = true;
            console.warn("[EqualizerWidget] animate frame error (logged once):", e);
          }
        }
        if (playing) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }

    // --- interaction ------------------------------------------------------------
    function canvasPoint(canvas, e) {
      const r = canvas.getBoundingClientRect();
      return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * (canvas === eq.canvas ? EQ_H : WAVE_H)];
    }

    function nearestBand(x) {
      let best = -1;
      let bestD = Infinity;
      bands.forEach((f, i) => {
        const d = Math.abs(fx(f) - x);
        if (d < bestD) { bestD = d; best = i; }
      });
      return bestD < 36 ? best : -1;
    }

    eq.canvas.addEventListener("pointerdown", (e) => {
      const [x, y] = canvasPoint(eq.canvas, e);
      const i = nearestBand(x);
      if (i < 0) return;
      dragBand = focusBand = i;
      eq.canvas.setPointerCapture(e.pointerId);
      gains[i] = yg(y);
      updateResponse();
    });
    eq.canvas.addEventListener("pointermove", (e) => {
      const [x, y] = canvasPoint(eq.canvas, e);
      if (dragBand >= 0) {
        gains[dragBand] = yg(y);
        updateResponse();
      } else {
        eq.canvas.style.cursor = nearestBand(x) >= 0 ? "ns-resize" : "default";
      }
    });
    const endDrag = () => {
      if (dragBand < 0) return;
      dragBand = -1;
      drawEq();
      commit();
    };
    eq.canvas.addEventListener("pointerup", endDrag);
    eq.canvas.addEventListener("pointercancel", endDrag);
    eq.canvas.addEventListener("dblclick", (e) => {
      const i = nearestBand(canvasPoint(eq.canvas, e)[0]);
      if (i < 0) return;
      gains[i] = 1;
      updateResponse();
      commit();
    });
    eq.canvas.addEventListener("keydown", (e) => {
      if (focusBand < 0) focusBand = 0;
      const step = e.shiftKey ? 0.1 : 0.02;
      if (e.key === "ArrowLeft") focusBand = Math.max(0, focusBand - 1);
      else if (e.key === "ArrowRight") focusBand = Math.min(bands.length - 1, focusBand + 1);
      else if (e.key === "ArrowUp") gains[focusBand] = Math.min(1, gains[focusBand] + step);
      else if (e.key === "ArrowDown") gains[focusBand] = Math.max(0, gains[focusBand] - step);
      else return;
      e.preventDefault();
      updateResponse();
      if (e.key === "ArrowUp" || e.key === "ArrowDown") commit();
    });
    eq.canvas.addEventListener("blur", () => { focusBand = -1; drawEq(); });

    wave.canvas.addEventListener("click", (e) => {
      if (!buffer) { input.click(); return; }
      const t = (canvasPoint(wave.canvas, e)[0] / W) * buffer.duration;
      if (playing) start(t);
      else { offset = t; drawWave(); }
    });

    playBtn.addEventListener("click", () => {
      if (!buffer) return;
      playing ? pause() : start(offset);
    });
    browse.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      loadFile(input.files[0]);
      input.value = "";
    });

    // drag and drop anywhere on the card
    let depth = 0;
    const isFile = (e) => Array.from(e.dataTransfer?.types || []).includes("Files");
    root.addEventListener("dragenter", (e) => {
      if (!isFile(e)) return;
      e.preventDefault();
      depth++;
      root.classList.add("dbeq-over");
    });
    root.addEventListener("dragover", (e) => {
      if (!isFile(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    });
    root.addEventListener("dragleave", () => {
      depth = Math.max(0, depth - 1);
      if (!depth) root.classList.remove("dbeq-over");
    });
    root.addEventListener("drop", (e) => {
      if (!isFile(e)) return;
      e.preventDefault();
      depth = 0;
      root.classList.remove("dbeq-over");
      const file = Array.from(e.dataTransfer.files).find(
        (f) => f.type.startsWith("audio/") || /\.(mp3|wav|ogg|m4a|aac|flac|webm)$/i.test(f.name)
      );
      loadFile(file);
    });

    // --- model sync -------------------------------------------------------------
    const onGains = () => {
      if (dragBand >= 0) return;
      setGains(model.get("gains"));
    };
    model.on("change:gains", onGains);
    model.on("change:bands", rebuildBands);

    rebuildBands();
    drawWave();

    return () => {
      cancelAnimationFrame(raf);
      model.off("change:gains", onGains);
      model.off("change:bands", rebuildBands);
      stopSource();
      if (audioCtx) audioCtx.close();
    };
  },
};
