// Main thread: wires the input, the token strip and the four views to the worker.
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


// ---------------------------------------------------------------- inside the networks: one strip per vector, at the selected token
const diverging = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);        // blue negative, red positive
const gateScale = d3.scaleSequential(d3.interpolateGreens).domain([0, 1]);          // gates: 0 -> white, 1 -> dark green
const readout = (() => { const el = document.createElement("div"); el.id = "hover-readout"; document.body.appendChild(el); return el; })();
const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));

function stripRow(container, name, sub, vec, { gate = false, tall = false } = {}) {
  const row = document.createElement("div"); row.className = "vrow";
  const label = document.createElement("div"); label.className = "vname"; label.innerHTML = `${name}<small>${sub}</small>`;
  const canvas = document.createElement("canvas"); canvas.width = vec.length; canvas.height = 1; if (tall) canvas.classList.add("tall");
  const stat = document.createElement("div"); stat.className = "vstat";
  let maxAbs = 1e-6, sum2 = 0, mean = 0;
  for (const v of vec) { maxAbs = Math.max(maxAbs, Math.abs(v)); sum2 += v * v; mean += v; }
  mean /= vec.length;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(vec.length, 1);
  for (let i = 0; i < vec.length; i++) {
    const rgb = d3.rgb(gate ? gateScale(vec[i]) : diverging(vec[i] / maxAbs)), o = 4 * i;
    img.data[o] = rgb.r; img.data[o + 1] = rgb.g; img.data[o + 2] = rgb.b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  stat.textContent = gate ? `mean ${mean.toFixed(2)}` : `norm ${Math.sqrt(sum2).toFixed(2)}, max ${fmt(maxAbs)}`;
  canvas.onmousemove = e => {
    const i = Math.min(vec.length - 1, Math.floor(e.offsetX / canvas.clientWidth * vec.length));
    readout.style.display = "block"; readout.style.left = (e.clientX + 12) + "px"; readout.style.top = (e.clientY + 12) + "px";
    readout.textContent = `${name}[${i}] = ${fmt(vec[i])}`;
  };
  canvas.onmouseleave = () => { readout.style.display = "none"; };
  row.append(label, canvas, stat); container.appendChild(row);
}
function lensRow(container, top) {                 // "logit lens": what the network would predict if it stopped here
  const el = document.createElement("div"); el.className = "lens";
  el.innerHTML = "if it stopped here: " + top.map(r => `<b>${show(r.text).replace(/</g, "&lt;")}</b> ${(100 * r.prob).toFixed(0)} %`).join(" · ");
  container.appendChild(el);
}
function opRow(container, text) { const el = document.createElement("div"); el.className = "vop"; el.innerHTML = text; container.appendChild(el); }
function sepRow(container, text) { const el = document.createElement("div"); el.className = "vsep"; el.innerHTML = text; container.appendChild(el); }
function listNeighbours(id, rows) {
  const ol = $(id); ol.replaceChildren();
  rows.forEach(r => { const li = document.createElement("li"); li.textContent = show(r.text) + " "; const s = document.createElement("span"); s.textContent = r.sim.toFixed(2); li.appendChild(s); ol.appendChild(li); });
}

function renderInside(m) {
  const I = m.inside, d = I.embedding.length, H = m.lstm.hidden;
  $("emb-note").textContent = `Selected token: ${JSON.stringify(I.text)} (id ${I.tokenId}). Each model has its own embedding table, so the same token has two vectors of ${d} numbers, learned independently by the two networks.`;
  const emb = $("emb-strips"); emb.replaceChildren();
  stripRow(emb, "CNN embedding", `E[${I.tokenId}], ${d}`, I.embedding);
  stripRow(emb, "LSTM embedding", `E[${I.tokenId}], ${d}`, I.lstmEmbedding);
  listNeighbours("nn-cnn", I.neighbours.cnn); listNeighbours("nn-lstm", I.neighbours.lstm);

  const c = $("cnn-inside"); c.replaceChildren();
  stripRow(c, "stream, layer 0", `h⁽⁰⁾ = embedding, ${d}`, I.cnn.layers[0].hIn);
  lensRow(c, I.cnn.lens[0]);
  I.cnn.layers.forEach((L, l) => {
    sepRow(c, `layer ${l + 1}: convolution over the stream at positions t−2, t−1, t`);
    stripRow(c, "content u", `${d}`, L.u);
    stripRow(c, "gate σ(g)", `${d}, between 0 and 1`, L.gate, { gate: true });
    const write = L.u.map((v, i) => v * L.gate[i]);
    stripRow(c, "written u ⊙ σ(g)", `what the layer adds`, write);
    const next = l + 1 < I.cnn.layers.length ? I.cnn.layers[l + 1].hIn : I.cnn.hFinal;
    stripRow(c, `stream, layer ${l + 1}`, `h⁽${l + 1}⁾ = h⁽${l}⁾ + written`, next);
    lensRow(c, I.cnn.lens[l + 1]);
  });
  sepRow(c, "LayerNorm: centre, scale, and re-weight each coordinate");
  stripRow(c, "z", `LayerNorm(h⁽⁴⁾), ${d}`, I.cnn.z);

  const s = $("lstm-inside"); s.replaceChildren();
  I.lstm.layers.forEach((L, l) => {
    sepRow(c === null ? s : s, `layer ${l + 1}`);
    stripRow(s, "input x", l === 0 ? `the embedding, ${L.x.length}` : `output of layer ${l}, ${L.x.length}`, L.x);
    stripRow(s, "previous output h<sub>t−1</sub>", `${H}`, L.hPrev);
    stripRow(s, "previous memory c<sub>t−1</sub>", `${H}`, L.cPrev);
    opRow(s, "from x and h<sub>t−1</sub>, four vectors:");
    stripRow(s, "candidate c̃", `tanh, ${H}`, L.candidate);
    stripRow(s, "forget gate f", `keep how much of c<sub>t−1</sub>`, L.forget, { gate: true });
    stripRow(s, "input gate i", `write how much of c̃`, L.input, { gate: true });
    stripRow(s, "output gate o", `show how much of the memory`, L.output, { gate: true });
    opRow(s, "c<sub>t</sub> = f ⊙ c<sub>t−1</sub> + i ⊙ c̃ &nbsp;&nbsp; h<sub>t</sub> = o ⊙ tanh(c<sub>t</sub>)");
    stripRow(s, "new memory c<sub>t</sub>", `${H}`, L.cell);
    stripRow(s, "output h<sub>t</sub>", `${H}`, L.h);
  });
  sepRow(s, "projection of the last output back to the embedding space");
  stripRow(s, "z", `W h<sub>t</sub> + b, ${I.lstm.z.length}`, I.lstm.z);

  const top = (model, name) => `${name}: z · E[${JSON.stringify(model.top[0].text)}] is the largest of the 8,000 dot products, so ${JSON.stringify(model.top[0].text)} gets ${(100 * model.top[0].prob).toFixed(1)} %`;
  $("logit-note").textContent = `For both models the 8,000 logits are the dot products of z with the 8,000 embedding vectors (the same table used at the input), and the softmax turns them into the distributions shown above. ${top(m.cnn, "CNN")}; ${top(m.lstm, "LSTM")}.`;
}


// ---------------------------------------------------------------- tokenization: pieces and the merge ladder
function renderTokenization(m) {
  const tz = m.inside.tokenization, strip = $("pieces"); strip.replaceChildren();
  tz.pieces.forEach((pc, i) => { const c = chip(pc.text, "static piece" + (i === tz.selected ? " selected" : "")); c.title = `${pc.nTokens} token${pc.nTokens > 1 ? "s" : ""}`; strip.appendChild(c); });
  const L = $("ladder"); L.replaceChildren();
  const ex = tz.explain;
  const row = (name, html) => { const r = document.createElement("div"); r.className = "lrow"; r.innerHTML = `<div class="lname">${name}</div><div>${html}</div>`; L.appendChild(r); };
  const esc = str => str.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  row("piece", `<span class="sym">${esc(show(ex.piece))}</span> (${ex.bytes.length} byte${ex.bytes.length > 1 ? "s" : ""})`);
  row("bytes", ex.bytes.map(b => `<span class="sym">${b.toString(16).padStart(2, "0")}</span>`).join(""));
  ex.steps.forEach((st, i) => {
    const syms = st.symbols.map(sym => `<span class="sym${sym === st.merged ? " merged" : ""}" title="${esc(sym)}">${esc(show(st.texts[st.symbols.indexOf(sym)]))}</span>`).join("");
    row(i === 0 ? "byte-level characters" : `merge ${i} <span class="muted">(rank ${st.rank})</span>`, syms);
  });
  row("tokens", ex.ids.map((id, i) => `<span class="sym">${esc(show(ex.steps[ex.steps.length - 1].texts[i]))}</span> = ${id}`).join(" &nbsp; "));
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

function render(m) {
  if (!m.T) { for (const id of ["tokens", "surprise-cnn", "surprise-lstm", "cnn-rf", "cnn-influence", "lstm-influence", "pieces", "ladder", "emb-strips", "cnn-inside", "lstm-inside"]) $(id).replaceChildren(); return; }
  renderStrips(m);
  renderBars("cnn-bars", m.cnn, m.actual, "cnn-actual");
  renderBars("lstm-bars", m.lstm, m.actual, "lstm-actual");
  renderHeatmaps(m);
  renderInfluence("cnn-influence", m.influence.cnn, m.tokens);
  renderInfluence("lstm-influence", m.influence.lstm, m.tokens);
  renderCurve(m);
  renderTokenization(m);
  renderInside(m);
  renderMap("map-cnn", m.inside.maps.cnn); renderMap("map-lstm", m.inside.maps.lstm);
  renderMemory(m);
}
