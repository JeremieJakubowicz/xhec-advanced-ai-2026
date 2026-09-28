// Web Worker: loads the tokenizer and the two models once, then answers analysis requests off the main thread.
// Each request is answered in two phases: first the main result (the forward passes and everything the page draws first),
// then, in deferred chunks that a newer request cancels, the intermediate logit lens, the ablations and the context curves.
import { ByteLevelBPE } from "./tokenizer.js";
import { parseSafetensors, LocalCNN, LSTMLM, logSoftmax, topk, nearestTokens, influenceCNN, influenceLSTM } from "./models.js";

const MAX_TOKENS = 64;
const WINDOW = 16;                                 // positions followed by the step-by-step diagrams, ending at the selected token
let tok, cnn, lstm, frequent = [], pca = {}, requestId = 0;

// two principal components of the embedding rows of the frequent tokens (power iteration with deflation)
function fitPCA(E, d, ids) {
  const n = ids.length, mean = new Float32Array(d);
  for (const id of ids) for (let i = 0; i < d; i++) mean[i] += E[id * d + i] / n;
  const X = new Float32Array(n * d);
  ids.forEach((id, r) => { for (let i = 0; i < d; i++) X[r * d + i] = E[id * d + i] - mean[i]; });
  const cov = new Float32Array(d * d);
  for (let r = 0; r < n; r++) for (let i = 0; i < d; i++) { const xi = X[r * d + i]; if (xi === 0) continue; for (let j = 0; j < d; j++) cov[i * d + j] += xi * X[r * d + j]; }
  const comps = [];
  for (let k = 0; k < 2; k++) {
    let v = new Float32Array(d).map((_, i) => Math.sin(i + k + 1));
    for (let it = 0; it < 60; it++) {
      const u = new Float32Array(d);
      for (let i = 0; i < d; i++) { let acc = 0; for (let j = 0; j < d; j++) acc += cov[i * d + j] * v[j]; u[i] = acc; }
      for (const c of comps) { let p = 0; for (let i = 0; i < d; i++) p += u[i] * c[i]; for (let i = 0; i < d; i++) u[i] -= p * c[i]; }
      let norm = 0; for (let i = 0; i < d; i++) norm += u[i] * u[i]; norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < d; i++) v[i] = u[i] / norm;
    }
    comps.push(v);
  }
  return { mean, comps, project(id) { const out = [0, 0]; for (let k = 0; k < 2; k++) for (let i = 0; i < d; i++) out[k] += (E[id * d + i] - mean[i]) * comps[k][i]; return out; } };
}

async function fetchBuffer(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

async function init() {
  postMessage({ type: "status", text: "loading the tokenizer…" });
  tok = new ByteLevelBPE(JSON.parse(new TextDecoder().decode(await fetchBuffer("models/tokenizer.json", "tokenizer"))));
  postMessage({ type: "status", text: "loading the CNN (7 MB)…" });
  cnn = new LocalCNN(parseSafetensors(await fetchBuffer("models/cnn.safetensors", "cnn")));
  postMessage({ type: "status", text: "loading the LSTM (12 MB)…" });
  lstm = new LSTMLM(parseSafetensors(await fetchBuffer("models/lstm.safetensors", "lstm")));
  try { frequent = JSON.parse(new TextDecoder().decode(await fetchBuffer("models/frequent.json", "frequent"))); } catch (e) { frequent = []; }
  const ids = frequent.map(([id]) => id);
  if (ids.length) { pca.cnn = fitPCA(cnn.E, cnn.d, ids); pca.lstm = fitPCA(lstm.E, lstm.d, ids); }
  postMessage({ type: "ready", receptiveField: cnn.receptiveField, hidden: lstm.hidden, maxTokens: MAX_TOKENS, window: WINDOW,
                cnnWeights: cnn.layers.map(L => ({ Wk: L.Wk, b: L.b })),         // the filters themselves (6 MB, once), so that the page can show any one of them
                thumbs: {                                                        // the big matrices as blocks: the two embedding tables, the LSTM's stacked gate weights
                  E: { cnn: thumb((r, c) => cnn.E[r * cnn.d + c], cnn.V, cnn.d), lstm: thumb((r, c) => lstm.E[r * lstm.d + c], lstm.V, lstm.d) },
                  lstm: lstm.layers.map(L => ({ ih: thumb((r, c) => L.Wih[r * L.inp + c], 4 * L.hidden, L.inp), hh: thumb((r, c) => L.Whh[r * L.hidden + c], 4 * L.hidden, L.hidden) })),
                  proj: thumb((r, c) => lstm.projW[r * lstm.hidden + c], lstm.d, lstm.hidden),   // the output projection, nn.Linear(hidden, d)
                } });
}

// a matrix as a small block: rows × cols sampled down to R × C values (not to scale), for the page to draw
function thumb(get, rows, cols, R = 96, C = 160) {
  R = Math.min(R, rows); C = Math.min(C, cols);
  const data = new Float32Array(R * C);
  for (let y = 0; y < R; y++) { const r = Math.floor(y * rows / R); for (let x = 0; x < C; x++) data[y * C + x] = get(r, Math.floor(x * cols / C)); }
  return { rows, cols, R, C, data };
}

function meanRows(arr, T, H) {                     // (T x H) -> T means
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) { let s = 0; for (let j = 0; j < H; j++) s += arr[t * H + j]; out[t] = s / H; }
  return out;
}

// forward pass; the surprise of every token; the prediction after `position` (top 10) and the top 3 at every position of the window,
// all from one pass over the logits
function analyseModel(model, ids, position, p0) {
  const T = ids.length, state = model.forward(ids);
  const logprobs = new Float32Array(T - 1), tops = [];
  let lpAt = null;
  for (let t = 0; t < T; t++) {
    if (t === T - 1 && t > position) continue;      // the last token, not selected: nothing follows it and it is outside the window
    const lp = logSoftmax(model.logitsAt(state, t));
    if (t < T - 1) logprobs[t] = lp[ids[t + 1]];
    if (t >= p0 && t <= position) tops.push(topk(lp, 3));
    if (t === position) lpAt = lp;
  }
  const actual = position + 1 < T ? ids[position + 1] : null;
  return { state, logprobs, lpAt, tops, top: topk(lpAt, 10), actualLogprob: actual === null ? null : lpAt[actual] };
}

onmessage = e => {
  const id = ++requestId;
  const { text: raw, position: wanted, ks } = e.data;
  const text = raw.trim().split(/\s+/).join(" ");                 // whitespace collapsed to one space, as in the training corpus
  const tokens = tok.tokenize(text);
  const truncated = tokens.length > MAX_TOKENS;
  const kept = tokens.slice(0, MAX_TOKENS), ids = kept.map(t => t.id), T = ids.length;
  if (T === 0) { postMessage({ type: "result", id, tokens: [], T: 0 }); return; }
  const position = Math.min(Math.max(wanted ?? Math.max(T - 2, 0), 0), T - 1);   // default: predict the last token, so there is an actual next token to compare with
  const p0 = Math.max(0, position - WINDOW + 1), t0 = performance.now();
  const c = analyseModel(cnn, ids, position, p0), l = analyseModel(lstm, ids, position, p0);
  const row = (arr, width, t) => arr.slice(t * width, (t + 1) * width);        // one position of a (T x width) array
  const decode = rows => rows.map(r => ({ ...r, text: tok.decode([r.id]) }));
  const p = position, d = cnn.d;
  const inside = {
    tokenId: ids[p], text: kept[p].text,
    embedding: row(cnn.E, d, ids[p]), lstmEmbedding: row(lstm.E, d, ids[p]),
    neighbours: { cnn: decode(nearestTokens(cnn, ids[p])), lstm: decode(nearestTokens(lstm, ids[p])) },
  };
  // embedding maps: frequent tokens + the text's tokens + the neighbours, projected on the two principal components
  inside.maps = {};
  for (const [name, model] of [["cnn", cnn], ["lstm", lstm]]) {
    if (!pca[name]) continue;
    const pts = [], seenIds = new Set();
    const add = (id, kind, count) => { if (seenIds.has(id) && kind === "frequent") return; seenIds.add(id); const [x, y] = pca[name].project(id); pts.push({ id, text: tok.decode([id]), x, y, kind, count }); };
    frequent.forEach(([id, count]) => add(id, "frequent", count));
    ids.forEach(id => add(id, "text"));
    inside.neighbours[name].forEach(nb => add(nb.id, "neighbour"));
    add(ids[p], "selected");
    inside.maps[name] = pts;
  }
  // step by step: every intermediate vector at every position (the diagrams pick their window). The "logit lens" of the CNN, what it
  // would predict from the stream after each layer: after the last layer it is the prediction itself, already computed at every
  // position; after the other layers it is computed now at the selected position and later at the others.
  const streams = [...c.state.record.map(r => r.hIn), c.state.hFinal], L = c.state.record.length, n = p - p0 + 1;
  const lensAt = (l, t) => decode(topk(logSoftmax(cnn.logitsFromStream(row(streams[l], d, t))), 3));
  const lens = streams.map((_, l) => Array.from({ length: n }, (_, i) => l === L ? decode(c.tops[i]) : null));
  for (let l = 0; l < L; l++) lens[l][n - 1] = lensAt(l, p);
  const lstmZ = []; for (let t = p0; t <= p; t++) lstmZ.push(lstm.projAt(l.state, t));
  const flow = {
    p0,
    cnn: { layers: c.state.record.map(r => ({ hIn: r.hIn, u: r.u, gate: r.gate })), hFinal: c.state.hFinal, z: c.state.h, lens },
    lstm: { layers: l.state.record.map(r => ({ x: r.x, nin: r.nin, input: r.input, forget: r.forget, candidate: r.candidate, output: r.output, cell: r.cell, h: r.h })), z: lstmZ, top: l.tops.map(decode) },
  };
  postMessage({
    type: "result", id, T, position, truncated, tokens: kept.map(t => ({ id: t.id, text: t.text })), inside, flow,
    actual: p + 1 < T ? { id: ids[p + 1], text: kept[p + 1].text } : null,
    cnn: { logprobs: c.logprobs, top: decode(c.top), actualLogprob: c.actualLogprob, gateMeans: c.state.gates.map(g => meanRows(g, T, cnn.d)) },
    lstm: { logprobs: l.logprobs, top: decode(l.top), actualLogprob: l.actualLogprob, forget: l.state.forget, cell: l.state.cell, hidden: lstm.hidden },
    ms: Math.round(performance.now() - t0),
  });

  // ---- phase 2: deferred chunks; a newer request makes them stop
  const later = fn => setTimeout(() => { if (id === requestId) fn(); }, 0);
  later(() => {
    for (let l = 0; l < L; l++) for (let t = p0; t < p; t++) lens[l][t - p0] = lensAt(l, t);
    postMessage({ type: "lens", id, lens });
    later(() => {
      const t1 = performance.now();
      postMessage({ type: "influence", id, influence: { cnn: influenceCNN(cnn, ids, p, c.lpAt), lstm: influenceLSTM(lstm, l.state, ids, p, l.lpAt, 16) } });
      // context curves, one k per chunk. The CNN cannot see beyond its window, so from there on its prediction is the full one;
      // the LSTM must be rerun from a blank memory for every k.
      const curves = { cnn: [], lstm: [] }, actual = p + 1 < T ? ids[p + 1] : null, kList = ks.filter(k => k <= p + 1);
      const curveRow = (k, lp) => ({ k, actualLogprob: actual === null ? null : lp[actual], top: decode(topk(lp, 5)) });
      const step = i => {
        if (i >= kList.length) { postMessage({ type: "curve", id, curves, ms: Math.round(performance.now() - t1) }); return; }
        const k = kList[i], ctx = ids.slice(p + 1 - k, p + 1);
        curves.cnn.push(curveRow(k, k >= cnn.receptiveField ? c.lpAt : logSoftmax(cnn.logitsAt(cnn.forward(ctx), k - 1))));
        curves.lstm.push(curveRow(k, k === p + 1 ? l.lpAt : logSoftmax(lstm.logitsAt(lstm.forward(ctx), k - 1))));
        later(() => step(i + 1));
      };
      step(0);
    });
  });
};

init().catch(err => postMessage({ type: "error", text: String(err) }));
