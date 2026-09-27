// Web Worker: loads the tokenizer and the two models once, then answers analysis requests off the main thread.
import { ByteLevelBPE } from "./tokenizer.js";
import { parseSafetensors, LocalCNN, LSTMLM, logSoftmax, topk } from "./models.js";

const MAX_TOKENS = 64;
let tok, cnn, lstm;

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
  return { state, logprobs, top, actualLogprob: actual === null ? null : lpAt[actual], curve };
}

onmessage = e => {
  const { text, position: wanted, ks } = e.data;
  const tokens = tok.tokenize(text);
  const truncated = tokens.length > MAX_TOKENS;
  const kept = tokens.slice(0, MAX_TOKENS), ids = kept.map(t => t.id), T = ids.length;
  if (T === 0) { postMessage({ type: "result", tokens: [], T: 0 }); return; }
  const position = Math.min(Math.max(wanted ?? Math.max(T - 2, 0), 0), T - 1);   // default: predict the last token, so there is an actual next token to compare with
  const t0 = performance.now();
  const c = analyseModel(cnn, ids, position, ks), l = analyseModel(lstm, ids, position, ks);
  const result = {
    type: "result", T, position, truncated, tokens: kept.map(t => ({ id: t.id, text: t.text })),
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
