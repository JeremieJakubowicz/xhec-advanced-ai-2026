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

function render(m) {
  if (!m.T) { for (const id of ["tokens", "surprise-cnn", "surprise-lstm", "cnn-rf"]) $(id).replaceChildren(); return; }
  renderStrips(m);
  renderBars("cnn-bars", m.cnn, m.actual, "cnn-actual");
  renderBars("lstm-bars", m.lstm, m.actual, "lstm-actual");
  renderHeatmaps(m);
  renderCurve(m);
}
