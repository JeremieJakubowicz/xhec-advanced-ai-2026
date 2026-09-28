// Main thread: wires the input, the token strips, the views, the two step-by-step diagrams and the guided tour to the worker.
const $ = id => document.getElementById(id);
const KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const LN2 = Math.log(2);
const bits = lp => -lp / LN2;
const show = s => s.replace(/\n/g, "⏎").replace(/ /g, "␣");
const surpriseColor = d3.scaleSequential(d3.interpolateBlues).domain([0, 12]).clamp(true);

const worker = new Worker("worker.js", { type: "module" });
let ready = false, pending = null, position = null, last = null, receptiveField = 9;

function request() {
  if (!ready) { pending = true; return; }
  $("status").textContent = "computing…";
  worker.postMessage({ text: $("text").value, position, ks: KS });
}
let timer = null;
$("text").addEventListener("input", () => { position = null; clearTimeout(timer); timer = setTimeout(request, 350); });

worker.onmessage = e => {
  const m = e.data;
  if (m.type === "status") { $("status").textContent = m.text; return; }
  if (m.type === "error") { $("status").textContent = "error: " + m.text; return; }
  if (m.type === "ready") { ready = true; receptiveField = m.receptiveField; $("rf").textContent = m.receptiveField; request(); return; }
  last = m; position = m.position;
  $("status").textContent = m.T ? `${m.T} tokens${m.truncated ? " (text cut at 64 tokens)" : ""}, prediction after token ${m.position + 1}` : "type something";
  $("timing").textContent = m.T ? `${m.ms} ms` : "";
  render(m);
};

// ---------------------------------------------------------------- token strips
function chip(text, cls = "") { const s = document.createElement("span"); s.className = "chip " + cls; s.textContent = show(text); return s; }

function renderStrips(m) {
  const strip = $("tokens"); strip.replaceChildren();
  m.tokens.forEach((t, i) => {
    const c = chip(t.text, i === m.position ? "selected" : i === m.position + 1 ? "next" : "");
    c.title = `token ${i + 1}, id ${t.id}`;
    c.onclick = () => { position = i; request(); };
    strip.appendChild(c);
  });
  for (const [id, model] of [["surprise-cnn", m.cnn], ["surprise-lstm", m.lstm]]) {
    const s = $(id); s.replaceChildren();
    m.tokens.forEach((t, i) => {
      const c = chip(t.text, "static");
      if (i > 0) { const b = bits(model.logprobs[i - 1]); c.style.background = surpriseColor(b); c.style.color = b > 7 ? "#fff" : "#000"; c.title = `${b.toFixed(1)} bits`; }
      else { c.style.color = "#aaa"; c.title = "first token: nothing to predict it from"; }
      s.appendChild(c);
    });
  }
  const rf = $("cnn-rf"); rf.replaceChildren();
  m.tokens.forEach((t, i) => {
    const inWindow = i <= m.position && i > m.position - receptiveField;
    rf.appendChild(chip(t.text, "static" + (inWindow ? " rf" : "") + (i === m.position ? " selected" : "")));
  });
}

// ---------------------------------------------------------------- next-token bars
function renderBars(svgId, model, actual, noteId) {
  const svg = d3.select("#" + svgId), W = 540, H = 230, m = { l: 118, r: 56, t: 4, b: 4 };          // logical size; CSS scales the viewBox
  svg.attr("viewBox", `0 0 ${W} ${H}`).selectAll("*").remove();
  const rows = model.top, y = d3.scaleBand().domain(rows.map((_, i) => i)).range([m.t, H - m.b]).padding(.18);
  const x = d3.scaleLinear().domain([0, Math.max(rows[0].prob, 0.05)]).range([m.l, W - m.r]);
  const g = svg.selectAll("g").data(rows).join("g").attr("transform", (_, i) => `translate(0,${y(i)})`);
  g.append("rect").attr("x", m.l).attr("height", y.bandwidth()).attr("width", d => Math.max(0, x(d.prob) - m.l))
    .attr("fill", d => actual && d.id === actual.id ? "#1b9e77" : "#d95f02").attr("opacity", .85);
  g.append("text").attr("class", "tok").attr("x", m.l - 6).attr("y", y.bandwidth() / 2).attr("dy", ".35em").attr("text-anchor", "end").text(d => show(d.text));
  g.append("text").attr("x", d => x(d.prob) + 4).attr("y", y.bandwidth() / 2).attr("dy", ".35em").attr("fill", "#555").text(d => (100 * d.prob).toFixed(1) + " %");
  const note = $(noteId);
  if (actual) {
    const lp = model.actualLogprob;
    note.textContent = `actual next token ${JSON.stringify(actual.text)}: ${(100 * Math.exp(lp)).toFixed(1)} %, surprise ${bits(lp).toFixed(1)} bits`;
  } else note.textContent = "(last token selected: no actual next token to compare with)";
}

// ---------------------------------------------------------------- heat maps on canvas
function paintHeatmap(canvas, rows, cols, value, color) {
  const ctx = canvas.getContext("2d"); canvas.width = cols; canvas.height = rows;
  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const rgb = d3.rgb(color(value(r, c))), o = 4 * (r * cols + c);
    img.data[o] = rgb.r; img.data[o + 1] = rgb.g; img.data[o + 2] = rgb.b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
const gateColor = d3.scaleSequential(d3.interpolateOranges).domain([0, 1]);
const forgetColor = d3.scaleSequential(d3.interpolateGreens).domain([0, 1]);

function renderHeatmaps(m) {
  paintHeatmap($("cnn-gates"), m.cnn.gateMeans.length, m.T, (r, c) => m.cnn.gateMeans[r][c], gateColor);
  $("cnn-gates-note").textContent = `rows: layers 1 to ${m.cnn.gateMeans.length} (top to bottom); columns: the ${m.T} tokens, left to right; colour: average gate value, 0 (white) to 1 (dark)`;
  const H = m.lstm.hidden;
  for (let l = 0; l < 2; l++) paintHeatmap($(`lstm-forget-${l}`), H, m.T, (r, c) => m.lstm.forget[l][c * H + r], forgetColor);
  $("lstm-forget-note").textContent = `rows: the ${H} memory slots of the layer; columns: the ${m.T} tokens, left to right; colour: forget gate, 0 (erase, white) to 1 (keep, dark)`;
}

// ---------------------------------------------------------------- context curve and slider
function renderCurve(m) {
  const sel = m.tokens[m.position].text;
  $("context-note").textContent = m.actual
    ? `Selected: token ${m.position + 1} (${JSON.stringify(sel)}). The token that actually follows is ${JSON.stringify(m.actual.text)}.`
    : `Selected: token ${m.position + 1} (${JSON.stringify(sel)}), the last one: no actual next token, the curves show the probability of each model's favourite token.`;
  const svg = d3.select("#curve"), W = 1120, H = 300, mg = { l: 56, r: 40, t: 14, b: 40 };
  svg.attr("viewBox", `0 0 ${W} ${H}`).selectAll("*").remove();
  const usesActual = m.actual !== null;
  const val = r => usesActual ? Math.exp(r.actualLogprob) : r.top[0].prob;
  const ks = m.cnn.curve.map(r => r.k);
  if (!ks.length) return;
  const x = d3.scaleLog().domain([1, Math.max(ks[ks.length - 1], 2)]).range([mg.l, W - mg.r]);
  const ymax = Math.max(...m.cnn.curve.map(val), ...m.lstm.curve.map(val), 0.02);
  const y = d3.scaleLinear().domain([0, ymax * 1.15]).range([H - mg.b, mg.t]);
  svg.append("g").attr("transform", `translate(0,${H - mg.b})`).call(d3.axisBottom(x).tickValues(ks).tickFormat(d3.format("d")));
  svg.append("g").attr("transform", `translate(${mg.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1%")));
  svg.append("text").attr("x", (mg.l + W - mg.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("fill", "#666").text("tokens of context shown to the model (k)");
  svg.append("text").attr("transform", `translate(14,${(mg.t + H - mg.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("fill", "#666")
    .text(usesActual ? "probability of the actual next token" : "probability of the favourite token");
  if (receptiveField <= ks[ks.length - 1]) {
    svg.append("line").attr("x1", x(receptiveField)).attr("x2", x(receptiveField)).attr("y1", mg.t).attr("y2", H - mg.b).attr("stroke", "#999").attr("stroke-dasharray", "3,3");
    svg.append("text").attr("x", x(receptiveField) + 4).attr("y", mg.t + 10).attr("fill", "#666").text("CNN window");
  }
  const line = d3.line().x(r => x(r.k)).y(r => y(val(r)));
  for (const [name, rows, color] of [["CNN", m.cnn.curve, "#d95f02"], ["LSTM", m.lstm.curve, "#1b9e77"]]) {
    svg.append("path").datum(rows).attr("d", line).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2);
    svg.selectAll(null).data(rows).join("circle").attr("cx", r => x(r.k)).attr("cy", r => y(val(r))).attr("r", 3.5).attr("fill", color);
    svg.append("text").attr("x", x(rows[rows.length - 1].k) + 6).attr("y", y(val(rows[rows.length - 1])) + 4).attr("fill", color).text(name);
  }
  const slider = $("k"); slider.max = ks.length - 1; slider.value = ks.length - 1;
  const marker = svg.append("line").attr("y1", mg.t).attr("y2", H - mg.b).attr("stroke", "#333").attr("stroke-width", 1);
  const update = () => {
    const i = +slider.value, k = ks[i]; $("k-value").textContent = k; marker.attr("x1", x(k)).attr("x2", x(k));
    for (const [id, rows] of [["k-cnn", m.cnn.curve], ["k-lstm", m.lstm.curve]]) {
      const ol = $(id); ol.replaceChildren();
      rows[i].top.forEach(r => { const li = document.createElement("li"); li.textContent = show(r.text) + " "; const s = document.createElement("span"); s.textContent = `${(100 * r.prob).toFixed(1)} %`; li.appendChild(s); ol.appendChild(li); });
    }
  };
  slider.oninput = update; update();
}

// ---------------------------------------------------------------- strips of numbers (one canvas per vector)
const diverging = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);        // blue negative, red positive
const gateScale = d3.scaleSequential(d3.interpolateGreens).domain([0, 1]);          // gates: 0 -> white, 1 -> dark green
const readout = (() => { const el = document.createElement("div"); el.id = "hover-readout"; document.body.appendChild(el); return el; })();
const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));
const rowOf = (arr, width, t) => arr.subarray(t * width, (t + 1) * width);        // one position of a (T x width) array
const norm = vec => { let s = 0; for (const v of vec) s += v * v; return Math.sqrt(s); };
const meanOf = vec => { let s = 0; for (const v of vec) s += v; return s / vec.length; };
const clip = (s, width) => { const n = Math.max(2, Math.floor(width / 6.7)); return s.length > n ? s.slice(0, n - 1) + "…" : s; };

function paintVector(ctx, vec, gate) {
  const img = ctx.createImageData(vec.length, 1);
  let maxAbs = 1e-6; for (const v of vec) maxAbs = Math.max(maxAbs, Math.abs(v));
  for (let i = 0; i < vec.length; i++) {
    const rgb = d3.rgb(gate ? gateScale(vec[i]) : diverging(vec[i] / maxAbs)), o = 4 * i;
    img.data[o] = rgb.r; img.data[o + 1] = rgb.g; img.data[o + 2] = rgb.b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0); return maxAbs;
}
function rasterURL(vec, gate = false) {              // a vector as a one-pixel-tall image, one pixel per coordinate (for the SVG diagrams)
  const cv = document.createElement("canvas"); cv.width = vec.length; cv.height = 1;
  paintVector(cv.getContext("2d"), vec, gate); return cv.toDataURL();
}

function stripRow(container, name, sub, vec, { gate = false, tall = false } = {}) {
  const row = document.createElement("div"); row.className = "vrow";
  const label = document.createElement("div"); label.className = "vname"; label.innerHTML = `${name}<small>${sub}</small>`;
  const canvas = document.createElement("canvas"); canvas.width = vec.length; canvas.height = 1; if (tall) canvas.classList.add("tall");
  const stat = document.createElement("div"); stat.className = "vstat";
  const maxAbs = paintVector(canvas.getContext("2d"), vec, gate);
  stat.textContent = gate ? `mean ${meanOf(vec).toFixed(2)}` : `norm ${norm(vec).toFixed(2)}, max ${fmt(maxAbs)}`;
  canvas.onmousemove = e => {
    const i = Math.min(vec.length - 1, Math.floor(e.offsetX / canvas.clientWidth * vec.length));
    readout.style.display = "block"; readout.style.left = (e.clientX + 12) + "px"; readout.style.top = (e.clientY + 12) + "px";
    readout.textContent = `${name.replace(/<[^>]+>/g, "")}[${i}] = ${fmt(vec[i])}`;
  };
  canvas.onmouseleave = () => { readout.style.display = "none"; };
  row.append(label, canvas, stat); container.appendChild(row);
}
function lensRow(container, top, label = "if it stopped here:") {           // "logit lens": what the network would predict from this vector
  const el = document.createElement("div"); el.className = "lens";
  el.innerHTML = label + " " + top.map(r => `<b>${show(r.text).replace(/</g, "&lt;")}</b> ${(100 * r.prob).toFixed(0)} %`).join(" · ");
  container.appendChild(el);
}
function opRow(container, text) { const el = document.createElement("div"); el.className = "vop"; el.innerHTML = text; container.appendChild(el); }
function sepRow(container, text) { const el = document.createElement("div"); el.className = "vsep"; el.innerHTML = text; container.appendChild(el); }
function listNeighbours(id, rows) {
  const ol = $(id); ol.replaceChildren();
  rows.forEach(r => { const li = document.createElement("li"); li.textContent = show(r.text) + " "; const s = document.createElement("span"); s.textContent = r.sim.toFixed(2); li.appendChild(s); ol.appendChild(li); });
}

function renderInside(m) {
  const I = m.inside, d = I.embedding.length;
  $("emb-note").textContent = `Selected token: ${JSON.stringify(I.text)} (id ${I.tokenId}). Each model has its own embedding table, so the same token has two vectors of ${d} numbers, learned independently by the two networks.`;
  const emb = $("emb-strips"); emb.replaceChildren();
  stripRow(emb, "CNN embedding", `E[${I.tokenId}], ${d}`, I.embedding);
  stripRow(emb, "LSTM embedding", `E[${I.tokenId}], ${d}`, I.lstmEmbedding);
  listNeighbours("nn-cnn", I.neighbours.cnn); listNeighbours("nn-lstm", I.neighbours.lstm);
  const top = (model, name) => `${name}: z · E[${JSON.stringify(model.top[0].text)}] is the largest of the 8,000 dot products, so ${JSON.stringify(model.top[0].text)} gets ${(100 * model.top[0].prob).toFixed(1)} %`;
  $("logit-note").textContent = `For both models the 8,000 logits are the dot products of z with the 8,000 embedding vectors (the same table used at the input), and the softmax turns them into the distributions shown at the top. ${top(m.cnn, "CNN")}; ${top(m.lstm, "LSTM")}.`;
}

// ---------------------------------------------------------------- embedding maps
function renderMap(svgId, pts) {
  const svg = d3.select("#" + svgId), W = 540, H = 380, mg = 14;
  svg.attr("viewBox", `0 0 ${W} ${H}`).selectAll("*").remove();
  if (!pts || !pts.length) return;
  const x = d3.scaleLinear().domain(d3.extent(pts, p => p.x)).nice().range([mg, W - mg]);
  const y = d3.scaleLinear().domain(d3.extent(pts, p => p.y)).nice().range([H - mg, mg]);
  const size = d3.scaleSqrt().domain([1, d3.max(pts, p => p.count || 1)]).range([1.5, 4]);
  const color = { frequent: "#bbb", text: "#d95f02", neighbour: "#1b9e77", selected: "#d95f02" };
  const order = { frequent: 0, text: 1, neighbour: 2, selected: 3 };
  const sorted = pts.slice().sort((a, b) => order[a.kind] - order[b.kind]);
  svg.selectAll("circle").data(sorted).join("circle").attr("cx", p => x(p.x)).attr("cy", p => y(p.y))
    .attr("r", p => p.kind === "selected" ? 7 : p.kind === "frequent" ? size(p.count || 1) : 4)
    .attr("fill", p => color[p.kind]).attr("opacity", p => p.kind === "frequent" ? .7 : .95).attr("stroke", p => p.kind === "selected" ? "#000" : "none")
    .append("title").text(p => show(p.text));
  const labelled = sorted.filter(p => p.kind !== "frequent" || (p.count && size(p.count) > 3.2));
  svg.selectAll("text").data(labelled).join("text").attr("x", p => x(p.x) + 5).attr("y", p => y(p.y) + 3)
    .attr("fill", p => p.kind === "frequent" ? "#888" : color[p.kind]).attr("font-weight", p => p.kind === "selected" ? 700 : 400).text(p => show(p.text));
}

// ---------------------------------------------------------------- which past tokens matter (leave-one-out)
const influenceColor = d3.scaleSequential(d3.interpolateOranges);
function renderInfluence(id, rows, tokens) {
  const strip = $(id); strip.replaceChildren();
  const max = Math.max(0.05, ...rows.map(r => r.bits));
  influenceColor.domain([0, max]);
  rows.forEach(r => {
    const c = chip(tokens[r.t].text, "static inf"); c.style.background = influenceColor(r.bits); c.style.color = r.bits > 0.6 * max ? "#fff" : "#000";
    c.title = `erasing token ${r.t + 1} changes the prediction by ${r.bits.toFixed(2)} bits`; strip.appendChild(c);
  });
}

// ---------------------------------------------------------------- the LSTM's memory over time
function renderMemory(m) {
  const H = m.lstm.hidden, T = m.T, C = m.lstm.cell[1], p = m.position;
  const slots = Array.from({ length: H }, (_, j) => j).sort((a, b) => Math.abs(C[p * H + b]) - Math.abs(C[p * H + a])).slice(0, 12);
  const svg = d3.select("#memory-traj"), W = 1120, Hh = 260, mg = { l: 48, r: 60, t: 12, b: 30 };
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).selectAll("*").remove();
  const x = d3.scaleLinear().domain([0, Math.max(T - 1, 1)]).range([mg.l, W - mg.r]);
  const ext = d3.max(slots, j => d3.max(d3.range(T), t => Math.abs(C[t * H + j]))) || 1;
  const y = d3.scaleLinear().domain([-ext, ext]).range([Hh - mg.b, mg.t]);
  svg.append("g").attr("transform", `translate(0,${Hh - mg.b})`).call(d3.axisBottom(x).ticks(Math.min(T, 16)).tickFormat(t => m.tokens[t] ? show(m.tokens[t].text) : ""));
  svg.append("g").attr("transform", `translate(${mg.l},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("line").attr("x1", mg.l).attr("x2", W - mg.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", "#ccc");
  svg.append("line").attr("x1", x(p)).attr("x2", x(p)).attr("y1", mg.t).attr("y2", Hh - mg.b).attr("stroke", "#d95f02").attr("stroke-dasharray", "3,3");
  const colors = d3.schemeTableau10.concat(["#999", "#555"]);
  slots.forEach((j, k) => {
    const line = d3.line().x(t => x(t)).y(t => y(C[t * H + j]));
    svg.append("path").datum(d3.range(T)).attr("d", line).attr("fill", "none").attr("stroke", colors[k]).attr("stroke-width", 1.6).attr("opacity", .9)
      .append("title").text(`memory slot ${j}`);
    svg.append("text").attr("x", W - mg.r + 4).attr("y", y(C[(T - 1) * H + j]) + 3).attr("fill", colors[k]).text(`#${j}`);
  });
}

// ---------------------------------------------------------------- step-by-step players (one per diagram)
const players = {};
function player(barId, nSteps, onStep, describe, interval) {
  const bar = $(barId); bar.replaceChildren();
  let i = nSteps - 1, timer = null, stopAt = nSteps - 1;                  // starts on the last step: the computation complete
  const btn = (label, title, fn) => { const b = document.createElement("button"); b.textContent = label; b.title = title; b.onclick = fn; bar.appendChild(b); return b; };
  const status = document.createElement("span"); status.className = "step-status";
  const set = j => { i = Math.max(0, Math.min(nSteps - 1, j)); onStep(i); status.textContent = describe(i); };
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } play.textContent = "▶ play"; };
  const run = (from, to) => { stop(); set(from); stopAt = to; play.textContent = "⏸ pause"; timer = setInterval(() => { if (i >= stopAt) stop(); else set(i + 1); }, interval); };
  btn("⏮", "first step", () => { stop(); set(0); });
  btn("‹", "previous step", () => { stop(); set(i - 1); });
  const play = btn("▶ play", "play the computation step by step", () => { if (timer) stop(); else run(i >= nSteps - 1 ? 0 : i, nSteps - 1); });
  btn("›", "next step", () => { stop(); set(i + 1); });
  btn("⏭", "last step", () => { stop(); set(nSteps - 1); });
  bar.appendChild(status);
  set(i);
  return { set: j => { stop(); set(j); }, stop, play: run, get index() { return i; } };
}

// ---------------------------------------------------------------- the CNN: a window of three sliding along the text, layer after layer
function renderCNNFlow(m) {
  const F = m.flow, d = m.inside.embedding.length, p = m.position, p0 = F.p0, n = p - p0 + 1, L = F.cnn.layers.length;
  const streams = [...F.cnn.layers.map(r => r.hIn), F.cnn.hFinal];                 // the stream after layer 0 (the embedding), 1, …, L
  const svg = d3.select("#cnn-flow"), W = 1120, left = 130, right = 14, c0 = Math.max(0, p0 - 2);   // c0: first drawn column, the two real tokens before the window
  const cw = Math.min(96, (W - left - right) / (p - c0 + 1)), cellW = cw - 10, cellH = 14, pitch = 60, top = 34;
  const rowY = r => top + r * pitch, zY = rowY(L) + pitch, predY = zY + cellH + 24, Hh = predY + 14;
  const x = t => left + (t - c0) * cw + 5;                                          // left edge of the cell at position t
  const wName = (j, l) => `W${"₀₁₂"[j]}⁽${l}⁾`;                                      // the weight matrix on the edge from position t−j into layer l
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).selectAll("*").remove();
  const part = name => svg.append("g").attr("data-part", name);
  const labels = ["embedding h⁽⁰⁾", ...d3.range(1, L + 1).map(l => `after layer ${l}: h⁽${l}⁾`)];
  const gTok = part("tokens");
  for (let t = c0; t <= p; t++) gTok.append("text").attr("class", "toklab" + (t === p ? " sel" : t < p0 ? " ctx" : "")).attr("x", x(t) + cellW / 2).attr("y", 18).attr("text-anchor", "middle").text(clip(show(m.tokens[t].text), cellW));
  if (c0 > 0) gTok.append("text").attr("class", "sublab").attr("x", left - 8).attr("y", 18).attr("text-anchor", "end").text(`… ${c0} earlier token${c0 > 1 ? "s" : ""}`);
  // one edge per (layer, position, offset j): the weight matrix W_j applied to the stream at t−j; drawn only where there is something to read
  const gLines = part("lines"), lines = [];
  for (let l = 1; l <= L; l++) for (let t = p0; t <= p; t++) for (let j = 0; j < 3; j++) {
    const s = t - j; if (s < 0) continue;
    lines.push({ l, t, s, j, x1: x(s) + cellW / 2, y1: rowY(l - 1) + cellH, x2: x(t) + cellW / 2, y2: rowY(l) });
  }
  const lineSel = gLines.selectAll("line").data(lines).join("line").attr("class", "flowline").attr("x1", q => q.x1).attr("y1", q => q.y1).attr("x2", q => q.x2).attr("y2", q => q.y2);
  lineSel.append("title").text(q => `${wName(q.j, q.l)} · h⁽${q.l - 1}⁾ at token ${q.s + 1}: one of the three terms of layer ${q.l} at token ${q.t + 1}`);
  for (let l = 1; l <= L; l++) gLines.append("text").attr("class", "sublab").attr("x", left - 8).attr("y", rowY(l) - 11).attr("text-anchor", "end").text(`edges: ${wName(2, l)} ${wName(1, l)} ${wName(0, l)}`)
    .append("title").text(`layer ${l} owns three matrices of 256 × 512 weights, one per offset (t−2, t−1, t), applied identically at every position`);
  const gCone = part("cone");
  const cellSel = [];
  for (let r = 0; r <= L; r++) {
    const g = part(r === 0 ? "emb" : `layer${r}`);
    g.append("text").attr("class", "rowlab").attr("x", left - 8).attr("y", rowY(r) + cellH / 2 + 4).attr("text-anchor", "end").text(labels[r] + (r === 0 ? " (256)" : ""));
    for (let t = c0; t < p0; t++) {                                                 // the two real tokens before the window: their actual values, dimmed
      const vec = rowOf(streams[r], d, t), cg = g.append("g").attr("class", "cell ctx").attr("transform", `translate(${x(t)},${rowY(r)})`);
      cg.append("image").attr("href", rasterURL(vec)).attr("width", cellW).attr("height", cellH).attr("preserveAspectRatio", "none").attr("opacity", .45);
      cg.append("rect").attr("class", "cellframe").attr("width", cellW).attr("height", cellH);
      cg.append("title").text(`${labels[r]}, token ${t + 1} (${show(m.tokens[t].text)}), before the window: norm ${norm(vec).toFixed(2)}`);
    }
    const cells = d3.range(p0, p + 1).map(t => ({ r, t, vec: rowOf(streams[r], d, t) }));
    const cg = g.selectAll("g.cell:not(.ctx)").data(cells).join("g").attr("class", "cell").attr("transform", c => `translate(${x(c.t)},${rowY(c.r)})`).style("cursor", "pointer")
      .on("click", (_, c) => players.cnn.set(c.r * n + (c.t - p0)));
    cg.append("image").attr("href", c => rasterURL(c.vec)).attr("width", cellW).attr("height", cellH).attr("preserveAspectRatio", "none");
    cg.append("rect").attr("class", "cellframe").attr("width", cellW).attr("height", cellH);
    cg.append("title").text(c => `${labels[c.r]}, token ${c.t + 1} (${show(m.tokens[c.t].text)}): norm ${norm(c.vec).toFixed(2)}`);
    cellSel.push(cg);
  }
  const gZ = part("z");
  gZ.append("text").attr("class", "rowlab").attr("x", left - 8).attr("y", zY + cellH / 2 + 4).attr("text-anchor", "end").text("LayerNorm → z");
  gZ.append("line").attr("class", "flowline").attr("x1", x(p) + cellW / 2).attr("y1", rowY(L) + cellH).attr("x2", x(p) + cellW / 2).attr("y2", zY);
  const zg = gZ.append("g").attr("transform", `translate(${x(p)},${zY})`);
  zg.append("image").attr("href", rasterURL(rowOf(F.cnn.z, d, p))).attr("width", cellW).attr("height", cellH).attr("preserveAspectRatio", "none");
  zg.append("rect").attr("class", "cellframe").attr("width", cellW).attr("height", cellH);
  zg.append("title").text(`z at token ${p + 1}: LayerNorm of h⁽${L}⁾, then 8,000 dot products with the embeddings`);
  const gPred = part("pred");
  gPred.append("text").attr("class", "rowlab").attr("x", left - 8).attr("y", predY + 4).attr("text-anchor", "end").text("predicted next token");
  const predSel = gPred.selectAll("text.predlab").data(d3.range(p0, p + 1)).join("text").attr("class", "predlab").attr("x", t => x(t) + cellW / 2).attr("y", predY + 4).attr("text-anchor", "middle")
    .text(t => clip(show(F.cnn.lens[L][t - p0][0].text), cellW));
  predSel.append("title").text(t => F.cnn.lens[L][t - p0].map(r => `${show(r.text)} ${(100 * r.prob).toFixed(0)} %`).join(", "));
  const gMark = part("marks");
  const bracket = gMark.append("rect").attr("class", "bracket").attr("rx", 4).attr("height", cellH + 8);
  const wLabels = [0, 1, 2].map(j => gMark.append("text").attr("class", "wlab").attr("text-anchor", "middle"));   // the names of the three active edges
  const target = gMark.append("rect").attr("class", "target").attr("rx", 3).attr("width", cellW + 6).attr("height", cellH + 6);
  const N = (L + 1) * n;                                                            // steps: layer-major, the order in which a convolution is computed
  const onStep = i => {
    const l = Math.floor(i / n), t = p0 + i % n;
    cellSel.forEach(sel => sel.attr("opacity", c => (c.r < l || (c.r === l && c.t <= t)) ? 1 : .15));
    lineSel.attr("class", q => "flowline" + (q.l === l && q.t === t ? " current" : (q.l < l || (q.l === l && q.t <= t)) ? "" : " future"));
    const cone = [];                                                                // every cell below that has contributed to the current one
    for (let r = 0; r < l; r++) for (let s = Math.max(c0, t - 2 * (l - r)); s <= t; s++) cone.push({ r, s });
    gCone.selectAll("rect").data(cone).join("rect").attr("class", "cone").attr("rx", 3).attr("x", c => x(c.s) - 3).attr("y", c => rowY(c.r) - 3).attr("width", cellW + 6).attr("height", cellH + 6);
    if (l >= 1) { const s0 = Math.max(0, t - 2); bracket.style("display", null).attr("x", x(s0) - 4).attr("y", rowY(l - 1) - 4).attr("width", x(t) + cellW + 4 - (x(s0) - 4)); }
    else bracket.style("display", "none");
    wLabels.forEach((lab, j) => {                                                   // W_j on the edge from t−j, staggered so that they do not overlap
      const s = t - j, on = l >= 1 && s >= 0;
      lab.style("display", on ? null : "none");
      if (on) lab.attr("x", (x(s) + x(t)) / 2 + cellW / 2).attr("y", (rowY(l - 1) + cellH + rowY(l)) / 2 + 4 + (1 - j) * 7).text(wName(j, l));
    });
    target.attr("x", x(t) - 3).attr("y", rowY(l) - 3);
    gZ.attr("opacity", l === L && t === p ? 1 : .15);
    predSel.attr("opacity", s => (l === L && s <= t) ? 1 : .15);
    renderCNNStep(m, l, t);
  };
  const describe = i => {
    const l = Math.floor(i / n), t = p0 + i % n, tk = JSON.stringify(m.tokens[t].text);
    if (l === 0) return `step ${i + 1} of ${N}: look up the embedding of token ${t + 1} ${tk}`;
    const terms = [2, 1, 0].filter(j => t - j >= 0).map(j => `${wName(j, l)} · token ${t - j + 1}`).join(" + ");
    const edge = t < 2 ? ` (${t === 0 ? "the two" : "one of the"} earlier slots fall${t === 0 ? "" : "s"} before the text: nothing to read there)` : "";
    return `step ${i + 1} of ${N}: layer ${l} at token ${t + 1} ${tk}: b + ${terms}${edge} → u, g → writes h⁽${l}⁾` + (l === L && t === p ? " → LayerNorm → 8,000 logits" : "");
  };
  players.cnn = player("cnn-controls", N, onStep, describe, 220);
}

function renderCNNStep(m, l, t) {
  const F = m.flow, d = m.inside.embedding.length, c = $("cnn-step"); c.replaceChildren();
  const tokAt = s => `token ${s + 1} ${JSON.stringify(m.tokens[s].text)}`;
  const streams = [...F.cnn.layers.map(r => r.hIn), F.cnn.hFinal];
  if (l === 0) {
    sepRow(c, `layer 0: the embedding of ${tokAt(t)}, row ${m.tokens[t].id} of the table E`);
    stripRow(c, "h⁽⁰⁾", `E[${m.tokens[t].id}], ${d}`, rowOf(streams[0], d, t));
    lensRow(c, F.cnn.lens[0][t - F.p0]);
    return;
  }
  const Lr = F.cnn.layers[l - 1];
  const wName = j => `W${"₀₁₂"[j]}⁽${l}⁾`;
  sepRow(c, `layer ${l} at ${tokAt(t)}: reads the stream h⁽${l - 1}⁾ at up to three positions, each through its own matrix`);
  for (let j = 2; j >= 0; j--) {
    const s = t - j;
    if (s < 0) { opRow(c, `${wName(j)} has nothing to read: position t−${j} is before the start of the text`); continue; }
    stripRow(c, `${wName(j)} · h⁽${l - 1}⁾ at ${j ? `t−${j}` : "t"}`, tokAt(s), rowOf(streams[l - 1], d, s));
  }
  opRow(c, `a = b⁽${l}⁾ + ${[2, 1, 0].filter(j => t - j >= 0).map(j => `${wName(j)} h${j ? `ₜ₋${"₀₁₂"[j]}` : "ₜ"}`).join(" + ")}, ${2 * d} numbers: the first ${d} are the content u, the last ${d} the gate g`);
  const u = rowOf(Lr.u, d, t), g = rowOf(Lr.gate, d, t);
  stripRow(c, "content u", `${d}`, u);
  stripRow(c, "gate σ(g)", `${d}, between 0 and 1`, g, { gate: true });
  stripRow(c, "written u ⊙ σ(g)", "what the layer adds", u.map((v, i) => v * g[i]));
  stripRow(c, `h⁽${l}⁾ at t`, `h⁽${l - 1}⁾ at t + written`, rowOf(streams[l], d, t));
  lensRow(c, F.cnn.lens[l][t - F.p0]);
  if (l === F.cnn.layers.length && t === m.position) {
    sepRow(c, "the last layer at the selected token: LayerNorm (centre, scale, re-weight each coordinate), then 8,000 dot products with the embeddings");
    stripRow(c, "z", `LayerNorm(h⁽${l}⁾), ${d}`, rowOf(F.cnn.z, d, t));
    lensRow(c, F.cnn.lens[l][t - F.p0], "prediction:");
  }
}

// ---------------------------------------------------------------- the LSTM: one cell applied at every token, passing (h, c) along
function renderLSTMFlow(m) {
  const F = m.flow, p = m.position, p0 = F.p0, n = p - p0 + 1, H = m.lstm.hidden, Ls = F.lstm.layers;
  const svg = d3.select("#lstm-flow"), W = 1120, left = 130, right = 14, cols = n + 1;   // one extra column on the left: the state carried in
  const cw = Math.min(96, (W - left - right) / cols), cellW = cw - 14, cellH = 14, boxH = 84;
  const xY = 34, lY = [xY + cellH + 34, xY + cellH + 34 + boxH + 34];
  const survY = lY[1] + boxH + 28, predY = survY + 20 + 26, Hh = predY + 14;
  const x = t => left + (t - p0 + 1) * cw + 7;                                     // left edge of column t; column p0−1 is the carried state
  const cY = y => y + 44, hY = y => y + 66;                                          // y of the memory and output strips inside a box
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).selectAll("*").remove();
  const defs = svg.append("defs");
  for (const [id, color] of [["arr", "#999"], ["arr-mem", "#1b9e77"], ["arr-cur", "#d95f02"]])
    defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", color);
  const part = name => svg.append("g").attr("data-part", name);
  const gTok = part("tokens");
  for (let t = p0; t <= p; t++) gTok.append("text").attr("class", "toklab" + (t === p ? " sel" : "")).attr("x", x(t) + cellW / 2).attr("y", 18).attr("text-anchor", "middle").text(clip(show(m.tokens[t].text), cellW));
  gTok.append("text").attr("class", "sublab").attr("x", x(p0 - 1) + cellW / 2).attr("y", 18).attr("text-anchor", "middle").text(p0 > 0 ? "carried in" : "start");
  const gX = part("x");
  gX.append("text").attr("class", "rowlab").attr("x", left - 8).attr("y", xY + cellH / 2 + 4).attr("text-anchor", "end").text("input x: the embedding");
  const xCells = gX.selectAll("g").data(d3.range(p0, p + 1)).join("g").attr("transform", t => `translate(${x(t)},${xY})`).style("cursor", "pointer").on("click", (_, t) => players.lstm.set(t - p0));
  xCells.append("image").attr("href", t => rasterURL(rowOf(Ls[0].x, Ls[0].nin, t))).attr("width", cellW).attr("height", cellH).attr("preserveAspectRatio", "none");
  xCells.append("rect").attr("class", "cellframe").attr("width", cellW).attr("height", cellH);
  xCells.append("title").text(t => `embedding of token ${t + 1} (${show(m.tokens[t].text)})`);
  // arrows: memory c and output h from step t−1 to step t, input from above; drawn before the boxes
  const gArrC = part("arrows-c"), gArrH = part("arrows-h"), gArrV = part("arrows-v"), arrows = [];
  for (let l = 0; l < 2; l++) for (let t = p0; t <= p; t++) {
    const y = lY[l];
    arrows.push({ kind: "c", t, el: gArrC.append("line").attr("x1", x(t - 1) + cellW).attr("y1", cY(y) + 6).attr("x2", x(t) - 1).attr("y2", cY(y) + 6) });
    arrows.push({ kind: "h", t, el: gArrH.append("line").attr("x1", x(t - 1) + cellW).attr("y1", hY(y) + 6).attr("x2", x(t) - 1).attr("y2", hY(y) + 6) });
    arrows.push({ kind: "v", t, el: gArrV.append("line").attr("x1", x(t) + cellW / 2).attr("y1", l === 0 ? xY + cellH : lY[0] + boxH).attr("x2", x(t) + cellW / 2).attr("y2", y - 1) });
  }
  const meters = [["f", "forget", "#1b9e77"], ["i", "input", "#d95f02"], ["o", "output", "#6b6b6b"]];
  const boxSel = [];
  for (let l = 0; l < 2; l++) {
    const g = part(`l${l + 1}`), y = lY[l], Lr = Ls[l];
    g.append("text").attr("class", "rowlab").attr("x", left - 8).attr("y", y + 12).attr("text-anchor", "end").text(`layer ${l + 1}`);
    g.append("text").attr("class", "sublab").attr("x", left - 8).attr("y", y + 26).attr("text-anchor", "end").text("gates f, i, o (mean of 512)");
    g.append("text").attr("class", "sublab").attr("x", left - 8).attr("y", cY(y) + 10).attr("text-anchor", "end").text("memory c (512)");
    g.append("text").attr("class", "sublab").attr("x", left - 8).attr("y", hY(y) + 10).attr("text-anchor", "end").text("output h (512)");
    const cg = g.append("g").attr("transform", `translate(${x(p0 - 1)},${y})`);   // the state carried into the window
    cg.append("rect").attr("class", "box carried").attr("width", cellW).attr("height", boxH).attr("rx", 4);
    const cPrev = p0 > 0 ? rowOf(Lr.cell, H, p0 - 1) : new Float32Array(H), hPrev = p0 > 0 ? rowOf(Lr.h, H, p0 - 1) : new Float32Array(H);
    cg.append("text").attr("class", "gl").attr("x", cellW / 2).attr("y", 22).attr("text-anchor", "middle").text(p0 > 0 ? `after token ${p0}` : "zeros");
    cg.append("image").attr("href", rasterURL(cPrev)).attr("x", 3).attr("y", cY(y) - y).attr("width", cellW - 6).attr("height", 12).attr("preserveAspectRatio", "none");
    cg.append("image").attr("href", rasterURL(hPrev)).attr("x", 3).attr("y", hY(y) - y).attr("width", cellW - 6).attr("height", 12).attr("preserveAspectRatio", "none");
    const cells = d3.range(p0, p + 1).map(t => ({ l, t }));
    const bg = g.selectAll("g.cell").data(cells).join("g").attr("class", "cell").attr("transform", c => `translate(${x(c.t)},${y})`).style("cursor", "pointer").on("click", (_, c) => players.lstm.set(c.t - p0));
    bg.append("rect").attr("class", "box").attr("width", cellW).attr("height", boxH).attr("rx", 4);
    meters.forEach(([letter, key, color], k) => {
      const my = 8 + k * 11, bw = cellW - 22;
      bg.append("text").attr("class", "gl").attr("x", 4).attr("y", my + 7).text(letter);
      bg.append("rect").attr("class", "meterbg").attr("x", 15).attr("y", my).attr("width", bw).attr("height", 7);
      bg.append("rect").attr("x", 15).attr("y", my).attr("height", 7).attr("fill", color).attr("width", c => bw * meanOf(rowOf(Lr[key], H, c.t)));
    });
    bg.append("image").attr("href", c => rasterURL(rowOf(Lr.cell, H, c.t))).attr("x", 3).attr("y", cY(y) - y).attr("width", cellW - 6).attr("height", 12).attr("preserveAspectRatio", "none");
    bg.append("image").attr("href", c => rasterURL(rowOf(Lr.h, H, c.t))).attr("x", 3).attr("y", hY(y) - y).attr("width", cellW - 6).attr("height", 12).attr("preserveAspectRatio", "none");
    bg.append("title").text(c => `layer ${l + 1}, token ${c.t + 1} (${show(m.tokens[c.t].text)}): forget ${meanOf(rowOf(Lr.forget, H, c.t)).toFixed(2)}, input ${meanOf(rowOf(Lr.input, H, c.t)).toFixed(2)}, output ${meanOf(rowOf(Lr.output, H, c.t)).toFixed(2)}; |c| = ${norm(rowOf(Lr.cell, H, c.t)).toFixed(1)}`);
    boxSel.push(bg);
  }
  // how much of the layer-2 memory as it was after step s is still present at the selected token: f(s+1) × … × f(p), mean over the slots
  const gSurv = part("surv"), Fg = Ls[1].forget;
  gSurv.append("text").attr("class", "rowlab").attr("x", left - 8).attr("y", survY + 6).attr("text-anchor", "end").text("memory that survives");
  gSurv.append("text").attr("class", "sublab").attr("x", left - 8).attr("y", survY + 17).attr("text-anchor", "end").text("to the selected token");
  gSurv.append("text").attr("class", "sublab").attr("x", left - 8).attr("y", survY + 28).attr("text-anchor", "end").text("f × … × f, layer 2, mean");
  const surv = d3.range(p0, p + 1).map(s => { let acc = 0; for (let j = 0; j < H; j++) { let prod = 1; for (let k = s + 1; k <= p; k++) prod *= Fg[k * H + j]; acc += prod; } return { s, v: acc / H }; });
  const sg = gSurv.selectAll("g").data(surv).join("g").attr("transform", r => `translate(${x(r.s)},${survY})`);
  sg.append("rect").attr("width", cellW).attr("height", 20).attr("rx", 3).attr("fill", r => forgetColor(r.v)).attr("stroke", "#c9c9c9");
  sg.append("text").attr("class", "survtxt").attr("x", cellW / 2).attr("y", 14).attr("text-anchor", "middle").attr("fill", r => r.v > .55 ? "#fff" : "#222").text(r => r.v.toFixed(2));
  sg.append("title").text(r => `of the memory after token ${r.s + 1}, a share ${r.v.toFixed(3)} (on average over the 512 slots) survives to token ${p + 1}`);
  const gPred = part("pred");
  gPred.append("text").attr("class", "rowlab").attr("x", left - 8).attr("y", predY + 4).attr("text-anchor", "end").text("predicted next token");
  const predSel = gPred.selectAll("text.predlab").data(d3.range(p0, p + 1)).join("text").attr("class", "predlab").attr("x", t => x(t) + cellW / 2).attr("y", predY + 4).attr("text-anchor", "middle")
    .text(t => clip(show(F.lstm.top[t - p0][0].text), cellW));
  predSel.append("title").text(t => F.lstm.top[t - p0].map(r => `${show(r.text)} ${(100 * r.prob).toFixed(0)} %`).join(", "));
  const gMark = part("marks"), targets = [0, 1].map(l => gMark.append("rect").attr("class", "target").attr("rx", 5).attr("width", cellW + 6).attr("height", boxH + 6).attr("y", lY[l] - 3));
  const onStep = i => {
    const t = p0 + i;
    xCells.attr("opacity", s => s <= t ? 1 : .15);
    boxSel.forEach(sel => sel.attr("opacity", c => c.t <= t ? 1 : .15));
    for (const a of arrows) a.el.attr("class", `arrow${a.kind === "c" ? " mem" : ""}${a.t === t ? " current" : a.t > t ? " future" : ""}`)
      .attr("marker-end", a.t === t ? "url(#arr-cur)" : a.kind === "c" ? "url(#arr-mem)" : "url(#arr)");
    targets.forEach(r => r.attr("x", x(t) - 3));
    predSel.attr("opacity", s => s <= t ? 1 : .15);
    gSurv.attr("opacity", t === p ? 1 : .15);
    renderLSTMStep(m, t);
  };
  const describe = i => { const t = p0 + i; return `step ${i + 1} of ${n}: token ${t + 1} ${JSON.stringify(m.tokens[t].text)} comes in; layer 1 then layer 2 read their previous h and c, update the memory, pass it on` + (t === p ? " → projection → 8,000 logits" : ""); };
  players.lstm = player("lstm-controls", n, onStep, describe, 420);
}

function renderLSTMStep(m, t) {
  const F = m.flow, H = m.lstm.hidden, box = $("lstm-step"); box.replaceChildren();
  F.lstm.layers.forEach((Lr, l) => {
    const s = document.createElement("div"); s.className = "strips"; box.appendChild(s);
    sepRow(s, `layer ${l + 1}, step ${t + 1} (token ${JSON.stringify(m.tokens[t].text)})`);
    stripRow(s, "input x", l === 0 ? `the embedding of the token, ${Lr.nin}` : `output of layer 1 at this step, ${Lr.nin}`, rowOf(Lr.x, Lr.nin, t));
    stripRow(s, "previous output h<sub>t−1</sub>", t > 0 ? `${H}` : "start of the text: zeros", t > 0 ? rowOf(Lr.h, H, t - 1) : new Float32Array(H));
    stripRow(s, "previous memory c<sub>t−1</sub>", t > 0 ? `${H}` : "start of the text: zeros", t > 0 ? rowOf(Lr.cell, H, t - 1) : new Float32Array(H));
    opRow(s, "from x and h<sub>t−1</sub>, four vectors:");
    stripRow(s, "candidate c̃", `tanh, ${H}`, rowOf(Lr.candidate, H, t));
    stripRow(s, "forget gate f", "keep how much of c<sub>t−1</sub>", rowOf(Lr.forget, H, t), { gate: true });
    stripRow(s, "input gate i", "write how much of c̃", rowOf(Lr.input, H, t), { gate: true });
    stripRow(s, "output gate o", "show how much of the memory", rowOf(Lr.output, H, t), { gate: true });
    opRow(s, "c<sub>t</sub> = f ⊙ c<sub>t−1</sub> + i ⊙ c̃ &nbsp;&nbsp; h<sub>t</sub> = o ⊙ tanh(c<sub>t</sub>)");
    stripRow(s, "new memory c<sub>t</sub>", `${H}`, rowOf(Lr.cell, H, t));
    stripRow(s, "output h<sub>t</sub>", `${H}`, rowOf(Lr.h, H, t));
  });
  const z = document.createElement("div"); z.className = "strips wide"; box.appendChild(z);
  sepRow(z, "projection of layer 2's output back to the embedding space, then 8,000 dot products with the embeddings");
  stripRow(z, "z", `W h<sub>t</sub> + b, ${F.lstm.z[t - F.p0].length}`, F.lstm.z[t - F.p0]);
  lensRow(z, F.lstm.top[t - F.p0], "prediction after this token:");
  renderCell(m, t);
}

// ---------------------------------------------------------------- one LSTM cell, wired as in Christopher Olah's figure, coloured with the current step (layer 2)
function renderCell(m, t) {
  const F = m.flow, H = m.lstm.hidden, Lr = F.lstm.layers[1];
  const f = meanOf(rowOf(Lr.forget, H, t)), i = meanOf(rowOf(Lr.input, H, t)), o = meanOf(rowOf(Lr.output, H, t));
  const svg = d3.select("#lstm-cell"), W = 560, Hh = 236;
  svg.attr("viewBox", `0 0 ${W} ${Hh}`).selectAll("*").remove();
  svg.append("defs").append("marker").attr("id", "cell-arr").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
    .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", "#666");
  const wire = (pts, cls = "wire") => svg.append("polyline").attr("class", cls).attr("points", pts.map(q => q.join(",")).join(" ")).attr("marker-end", "url(#cell-arr)");
  const dot = (x, y, cls = "") => svg.append("circle").attr("class", "dot " + cls).attr("cx", x).attr("cy", y).attr("r", 3);
  const op = (x, y, label) => { svg.append("circle").attr("class", "node").attr("cx", x).attr("cy", y).attr("r", 13); svg.append("text").attr("class", "oplab").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle").text(label); };
  const gate = (x, y, label, sub, value) => {
    svg.append("rect").attr("class", "node").attr("x", x - 22).attr("y", y - 15).attr("width", 44).attr("height", 30).attr("rx", 5).attr("fill", value === null ? "#fff" : gateScale(value));
    const ink = value !== null && value > .55 ? "#fff" : "#222";
    svg.append("text").attr("class", "gatelab").attr("x", x).attr("y", y - 2).attr("text-anchor", "middle").attr("fill", ink).text(label);
    svg.append("text").attr("class", "gateval").attr("x", x).attr("y", y + 10).attr("text-anchor", "middle").attr("fill", ink).text(sub);
  };
  const cY = 40, gY = 150, hY = 204, xF = 150, xI = 220, xC = 320, xO = 470, xAdd = 270, xTanh = 400;
  const label = (x, y, txt, anchor = "start", cls = "lab") => svg.append("text").attr("class", cls).attr("x", x).attr("y", y).attr("text-anchor", anchor).text(txt);
  // the memory line along the top: × forget, + (input ⊙ candidate), then on to the next step
  wire([[46, cY], [xF - 14, cY]], "wire mem"); wire([[xF + 13, cY], [xAdd - 14, cY]], "wire mem"); wire([[xAdd + 13, cY], [522, cY]], "wire mem");
  label(40, cY + 4, "c t−1", "end"); label(528, cY + 4, "c t");
  op(xF, cY, "×"); op(xAdd, cY, "+"); op(xAdd, 100, "×"); op(xTanh, 96, "tanh"); op(xTanh, 140, "×");
  // gates and the candidate, all read from the same [x t, h t−1]
  gate(xF, gY, "σ", `f ${f.toFixed(2)}`, f); gate(xI, gY, "σ", `i ${i.toFixed(2)}`, i); gate(xC, gY, "tanh", "c̃", null); gate(xO, gY, "σ", `o ${o.toFixed(2)}`, o);
  wire([[46, hY], [xO, hY]], "wire bus"); label(40, hY - 2, "h t−1", "end"); label(40, hY + 12, "x t", "end");
  for (const x of [xF, xI, xC, xO]) { dot(x, hY); wire([[x, hY], [x, gY + 16]]); }
  wire([[xF, gY - 16], [xF, cY + 14]]);                                           // f → ×
  wire([[xI, gY - 16], [xI, 100], [xAdd - 14, 100]]);                              // i → ×
  wire([[xC, gY - 16], [xC, 100], [xAdd + 14, 100]]);                              // c̃ → ×
  wire([[xAdd, 87], [xAdd, cY + 14]]);                                             // i ⊙ c̃ → +
  dot(xTanh, cY, "mem"); wire([[xTanh, cY], [xTanh, 82]]);                         // c t → tanh
  wire([[xTanh, 109], [xTanh, 126]]);                                              // tanh(c t) → ×
  wire([[xO, gY - 16], [xO, 140], [xTanh + 14, 140]]);                             // o → ×
  wire([[xTanh - 13, 140], [46, 140]]); label(40, 144, "h t", "end");              // h t leaves: to the next step, and up to the next layer
  label(xTanh - 20, 128, "h t = o ⊙ tanh(c t)", "end", "eq");
  label(xAdd + 18, 84, "c t = f ⊙ c t−1 + i ⊙ c̃", "start", "eq");
  svg.selectAll("text.lab, text.eq").each(function () {                             // " t", " t−1" as subscripts
    const el = d3.select(this), txt = el.text(); el.text("");
    txt.split(/( t−1| t)(?=[ )=]|$)/).forEach(part => { if (part === " t" || part === " t−1") el.append("tspan").attr("baseline-shift", "sub").attr("font-size", "8px").text(part.trim()); else if (part) el.append("tspan").text(part); });
  });
}

// ---------------------------------------------------------------- the guided tour: fifteen steps through the two architectures
function focus(svgId, parts) {                     // dim every part of a diagram except the ones named (null: show all)
  d3.select("#" + svgId).selectAll("g[data-part]").attr("opacity", function () { return !parts || parts.includes(this.dataset.part) ? 1 : .1; });
}
const focusAll = () => { focus("cnn-flow", null); focus("lstm-flow", null); };
const goTo = (id, quiet) => { if (!quiet) $(id).scrollIntoView({ behavior: "smooth", block: "center" }); };
const cnnStep = (m, l, t) => l * (m.position - m.flow.p0 + 1) + (t - m.flow.p0);
const CNN_ALL = ["tokens", "emb", "layer1", "layer2", "layer3", "layer4", "lines", "cone", "marks"];
const TOUR = [
  { title: "Tokens", text: "The text is cut into tokens by the byte-level BPE tokenizer trained on Hugo in session 1: a vocabulary of 8,000 pieces, each with an id. Both networks read the same ids. Click a token at any time: everything on this page is recomputed for the prediction after it.",
    run: (m, q) => { focusAll(); goTo("tokens", q); } },
  { title: "The embedding", text: "Each id selects one row of the embedding table E: 256 learned numbers, the same wherever the token appears in the text. Nothing says what they mean; training moves them until tokens used in the same way end up close to each other. The maps show the geometry that came out of it.",
    run: (m, q) => { focus("cnn-flow", ["tokens", "emb"]); players.cnn.set(cnnStep(m, 0, m.position)); goTo("emb-strips", q); } },
  { title: "One layer of the CNN reads three positions", text: "Layer 1, at token t, reads the stream at t−2, t−1 and t (the dashed bracket), each through its own matrix, W₂, W₁, W₀: the three edges. Their sum, plus a bias, gives a content u and a gate g. That is all a convolution is: three matrices, the same at every position. At the start of the text only the edges that have something to read exist.",
    run: (m, q) => { focus("cnn-flow", ["tokens", "emb", "layer1", "lines", "marks"]); players.cnn.set(cnnStep(m, 1, m.position)); goTo("cnn-flow", q); } },
  { title: "The gate and the residual addition", text: "σ(g), between 0 and 1, decides how much of u gets written; the layer adds u ⊙ σ(g) to the stream instead of replacing it, so what the embedding said is still there underneath. Below the diagram: the actual numbers of this step, one strip per vector, and what the network would already predict from them.",
    run: (m, q) => { focus("cnn-flow", ["tokens", "emb", "layer1", "lines", "marks"]); players.cnn.set(cnnStep(m, 1, m.position)); goTo("cnn-step", q); } },
  { title: "The window slides", text: "The same window and the same weights at every position of the text. No position waits for another: a whole layer is computed everywhere at once, which is why the CNN is fast to train and to run.",
    run: (m, q) => { focus("cnn-flow", ["tokens", "emb", "layer1", "lines", "marks"]); const a = cnnStep(m, 1, m.flow.p0), b = cnnStep(m, 1, m.position); if (q) players.cnn.set(b); else players.cnn.play(a, b); goTo("cnn-flow", q); } },
  { title: "Four layers: the receptive field", text: "Stack four such layers and a position sees 1 + 2 × 4 = 9 tokens (the light orange cells): its receptive field. Nothing further back can reach the prediction, ever; hence the context curve that goes flat at nine tokens.",
    run: (m, q) => { focus("cnn-flow", CNN_ALL); players.cnn.set(cnnStep(m, 4, m.position)); goTo("cnn-flow", q); } },
  { title: "From the stream to the distribution", text: "LayerNorm rescales the last stream vector into z; its dot products with the 8,000 embedding rows (the same table as at the input) are the logits, and the softmax turns them into the distribution at the top of the page. The last row shows the favourite token at every position of the window.",
    run: (m, q) => { focus("cnn-flow", ["tokens", "layer4", "z", "pred"]); players.cnn.set(cnnStep(m, 4, m.position)); goTo("cnn-flow", q); } },
  { title: "The LSTM: one cell", text: "A recurrent network has a single cell, applied at every token in turn. At step t it receives the input of the moment, x, together with its own previous output h and memory c, carried in from the step before (the arrows from the left). At the start of the text, h and c are zeros.",
    run: (m, q) => { focus("lstm-flow", ["tokens", "x", "l1", "arrows-c", "arrows-h", "arrows-v", "marks"]); players.lstm.set(0); goTo("lstm-flow", q); } },
  { title: "Three gates", text: "From x and h the cell computes a candidate c̃ and three gates, sigmoids between 0 and 1: f (forget) says how much of the old memory to keep, i (input) how much of the candidate to write, o (output) how much of the memory to show. The bars in each box are their means over the 512 slots.",
    run: (m, q) => { focus("lstm-flow", ["tokens", "x", "l1", "marks"]); players.lstm.set(0); goTo("lstm-flow", q); } },
  { title: "Updating the memory", text: "c = f ⊙ c_previous + i ⊙ c̃: what survived the forget gate, plus what the input gate let in. The thick green arrow is that memory handed to the next step, the only road from the past to the present.",
    run: (m, q) => { focus("lstm-flow", ["tokens", "l1", "arrows-c", "marks"]); players.lstm.set(0); goTo("lstm-flow", q); } },
  { title: "The output", text: "h = o ⊙ tanh(c): a filtered view of the memory. It goes two ways: to the next step, as the previous output, and up to layer 2, as its input.",
    run: (m, q) => { focus("lstm-flow", ["tokens", "l1", "arrows-h", "arrows-v", "l2", "marks"]); players.lstm.set(0); goTo("lstm-flow", q); } },
  { title: "Step after step", text: "Position t cannot start before position t−1 is finished: the steps are sequential, and that is the price of recurrence. Watch the state travel along the text, one token at a time.",
    run: (m, q) => { focusAll(); const n = m.position - m.flow.p0 + 1; if (q) players.lstm.set(n - 1); else players.lstm.play(0, n - 1); goTo("lstm-flow", q); } },
  { title: "Layer 2", text: "The outputs of layer 1 are the inputs of a second cell, with its own gates and its own memory. Its output at the selected token is what will be turned into the prediction.",
    run: (m, q) => { focus("lstm-flow", ["tokens", "l1", "arrows-v", "l2", "marks"]); players.lstm.set(m.position - m.flow.p0); goTo("lstm-flow", q); } },
  { title: "How far the memory reaches", text: "Whatever was in the memory after step s reaches the selected token multiplied by f(s+1) × … × f(p), one forget gate per step in between. The last row shows that product, averaged over the slots: a long reach needs gates that stay close to 1. That is what the gates buy over a plain RNN, and what training has to learn.",
    run: (m, q) => { focus("lstm-flow", ["tokens", "l2", "arrows-c", "surv"]); players.lstm.set(m.position - m.flow.p0); goTo("lstm-flow", q); } },
  { title: "Projection and prediction", text: "z = W h + b brings the 512 numbers back to 256; then the same 8,000 dot products and the same softmax as for the CNN. Both networks end the same way; they differ in how they build z: a fixed window of nine tokens, or a memory carried along step by step.",
    run: (m, q) => { focus("lstm-flow", ["tokens", "l2", "pred"]); players.lstm.set(m.position - m.flow.p0); goTo("lstm-step", q); } },
];
let tourIndex = -1;
function showTour(i, quiet = false) {
  if (!last || !last.T) return;
  tourIndex = Math.max(0, Math.min(TOUR.length - 1, i));
  const step = TOUR[tourIndex];
  $("tour").hidden = false; $("tour-title").textContent = `${tourIndex + 1}. ${step.title}`; $("tour-text").textContent = step.text;
  $("tour-pos").textContent = `${tourIndex + 1} / ${TOUR.length}`;
  $("tour-prev").disabled = tourIndex === 0; $("tour-next").textContent = tourIndex === TOUR.length - 1 ? "done" : "›";
  step.run(last, quiet);
}
function closeTour() { tourIndex = -1; $("tour").hidden = true; focusAll(); }
$("tour-start").onclick = () => showTour(0);
$("tour-prev").onclick = () => showTour(tourIndex - 1);
$("tour-next").onclick = () => tourIndex === TOUR.length - 1 ? closeTour() : showTour(tourIndex + 1);
$("tour-close").onclick = closeTour;

// ---------------------------------------------------------------- everything, on each result
function render(m) {
  Object.values(players).forEach(pl => pl.stop());
  if (!m.T) {
    for (const id of ["tokens", "surprise-cnn", "surprise-lstm", "cnn-rf", "cnn-influence", "lstm-influence", "emb-strips", "cnn-step", "lstm-step", "cnn-controls", "lstm-controls"]) $(id).replaceChildren();
    return;
  }
  renderStrips(m);
  renderBars("cnn-bars", m.cnn, m.actual, "cnn-actual");
  renderBars("lstm-bars", m.lstm, m.actual, "lstm-actual");
  renderHeatmaps(m);
  renderInfluence("cnn-influence", m.influence.cnn, m.tokens);
  renderInfluence("lstm-influence", m.influence.lstm, m.tokens);
  renderCurve(m);
  renderInside(m);
  renderMap("map-cnn", m.inside.maps.cnn); renderMap("map-lstm", m.inside.maps.lstm);
  renderCNNFlow(m);
  renderLSTMFlow(m);
  renderMemory(m);
  if (tourIndex >= 0) showTour(tourIndex, true);   // keep the tour's focus on the new result, without scrolling or replaying
}
