// Web Worker: loads the tokenizer and the two models once, then answers analysis requests off the main thread.
import { ByteLevelBPE } from "./tokenizer.js";
import { parseSafetensors, LocalCNN, LSTMLM, logSoftmax, topk, nearestTokens, influenceCNN, influenceLSTM } from "./models.js";

const MAX_TOKENS = 64;
let tok, cnn, lstm, frequent = [], pca = {};

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
  postMessage({ type: "ready", receptiveField: cnn.receptiveField, hidden: lstm.hidden, maxTokens: MAX_TOKENS });
}

function meanRows(arr, T, H) {                     // (T x H) -> T means
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) { let s = 0; for (let j = 0; j < H; j++) s += arr[t * H + j]; out[t] = s / H; }
  return out;
}

function analyseModel(model, ids, position, ks) {
  const T = ids.length, state = model.forward(ids);
  const logprobs = new Float32Array(T - 1);
  for (let t = 0; t < T - 1; t++) logprobs[t] = logSoftmax(model.logitsAt(state, t))[ids[t + 1]];
  const lpAt = logSoftmax(model.logitsAt(state, position));
  const actual = position + 1 < T ? ids[position + 1] : null;
  const top = topk(lpAt, 10).map(r => ({ ...r, text: tok.decode([r.id]) }));
  const curve = [];
  for (const k of ks) {
    if (k > position + 1) break;
    const ctx = ids.slice(position + 1 - k, position + 1);
    const lp = logSoftmax(model.logitsAt(model.forward(ctx), ctx.length - 1));
    curve.push({ k, actualLogprob: actual === null ? null : lp[actual], top: topk(lp, 5).map(r => ({ ...r, text: tok.decode([r.id]) })) });
  }
  return { state, logprobs, top, actualLogprob: actual === null ? null : lpAt[actual], curve, lpAt };
}

onmessage = e => {
  const { text: raw, position: wanted, ks } = e.data;
  const text = raw.trim().split(/\s+/).join(" ");                 // whitespace collapsed to one space, as in the training corpus
  const tokens = tok.tokenize(text);
  const truncated = tokens.length > MAX_TOKENS;
  const kept = tokens.slice(0, MAX_TOKENS), ids = kept.map(t => t.id), T = ids.length;
  if (T === 0) { postMessage({ type: "result", tokens: [], T: 0 }); return; }
  const position = Math.min(Math.max(wanted ?? Math.max(T - 2, 0), 0), T - 1);   // default: predict the last token, so there is an actual next token to compare with
  const t0 = performance.now();
  const c = analyseModel(cnn, ids, position, ks), l = analyseModel(lstm, ids, position, ks);
  const row = (arr, width, t) => arr.slice(t * width, (t + 1) * width);        // one position of a (T x width) array
  const p = position, d = cnn.d, H = lstm.hidden;
  const inside = {
    tokenId: ids[p], text: kept[p].text,
    embedding: row(cnn.E, d, ids[p]),
    neighbours: { cnn: nearestTokens(cnn, ids[p]).map(n => ({ ...n, text: tok.decode([n.id]) })),
                  lstm: nearestTokens(lstm, ids[p]).map(n => ({ ...n, text: tok.decode([n.id]) })) },
    lstmEmbedding: row(lstm.E, d, ids[p]),
    cnn: {
      layers: c.state.record.map(r => ({ hIn: row(r.hIn, d, p), u: row(r.u, d, p), gate: row(r.gate, d, p) })),
      hFinal: row(c.state.hFinal, d, p), z: row(c.state.h, d, p),
    },
    lstm: {
      layers: l.state.record.map(r => ({
        x: row(r.x, r.nin, p), hPrev: p > 0 ? row(r.h, H, p - 1) : new Float32Array(H), cPrev: p > 0 ? row(r.cell, H, p - 1) : new Float32Array(H),
        input: row(r.input, H, p), forget: row(r.forget, H, p), candidate: row(r.candidate, H, p), output: row(r.output, H, p),
        cell: row(r.cell, H, p), h: row(r.h, H, p) })),
      z: lstm.projAt(l.state, p),
    },
  };
  // tokenization of the piece that contains the selected token
  const pieceTexts = tok.pieces(text), pieces = [], selectedPiece = { index: 0 };
  let seen = 0;
  pieceTexts.forEach((pt, i) => { const n = tok.encode(pt).length; if (seen <= p && p < seen + n) selectedPiece.index = i; pieces.push({ text: pt, nTokens: n, first: seen }); seen += n; });
  const ex = tok.explain(pieceTexts[selectedPiece.index] || "");
  inside.tokenization = { pieces, selected: selectedPiece.index, explain: { ...ex, steps: ex.steps.map(st => ({ ...st, texts: st.symbols.map(sym => tok.decodeToken(sym)) })) } };
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
  const lens = [...c.state.record.map(r => row(r.hIn, d, p)), row(c.state.hFinal, d, p)].map(h => topk(logSoftmax(cnn.logitsFromStream(h)), 3).map(r => ({ ...r, text: tok.decode([r.id]) })));
  inside.cnn.lens = lens;                                                     // top-3 after layer 0 (the embedding), 1, 2, 3, 4
  const influence = { cnn: influenceCNN(cnn, ids, p, c.lpAt), lstm: influenceLSTM(lstm, l.state, ids, p, l.lpAt, 16) };
  const result = {
    type: "result", T, position, truncated, tokens: kept.map(t => ({ id: t.id, text: t.text })), inside, influence,
    actual: position + 1 < T ? { id: ids[position + 1], text: kept[position + 1].text } : null,
    cnn: { logprobs: c.logprobs, top: c.top, actualLogprob: c.actualLogprob, curve: c.curve,
           gateMeans: c.state.gates.map(g => meanRows(g, T, cnn.d)) },
    lstm: { logprobs: l.logprobs, top: l.top, actualLogprob: l.actualLogprob, curve: l.curve,
            forget: l.state.forget, cell: l.state.cell, hidden: lstm.hidden },
    ms: Math.round(performance.now() - t0),
  };
  postMessage(result, [...result.lstm.forget.map(a => a.buffer), ...result.lstm.cell.map(a => a.buffer)]);
};

init().catch(err => postMessage({ type: "error", text: String(err) }));
