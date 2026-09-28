// The two session-2 language models in plain JavaScript: the local gated CNN and the two-layer LSTM,
// with every intermediate value the site wants to show (gates, memory cells, per-position logits).
// Weights come from safetensors files written by tools/export_companion.py (float16).

// ---------------------------------------------------------------- safetensors (float16 -> Float32Array)
function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;                      // subnormal
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const HALF = new Float32Array(65536);
for (let i = 0; i < 65536; i++) HALF[i] = halfToFloat(i);

export function parseSafetensors(buffer) {
  const view = new DataView(buffer);
  const n = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, n)));
  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (info.dtype !== "F16") throw new Error(`unsupported dtype ${info.dtype} for ${name}`);
    const [a, b] = info.data_offsets;
    const u16 = new Uint16Array(buffer, 8 + n + a, (b - a) / 2);
    const f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) f32[i] = HALF[u16[i]];
    tensors[name] = { shape: info.shape, data: f32 };
  }
  return tensors;
}

// ---------------------------------------------------------------- small numeric helpers
const sigmoid = x => 1 / (1 + Math.exp(-x));
function dot(a, ao, b, bo, n) { let s = 0; for (let i = 0; i < n; i++) s += a[ao + i] * b[bo + i]; return s; }

export function logSoftmax(logits) {
  let m = -Infinity; for (const v of logits) if (v > m) m = v;
  let z = 0; for (const v of logits) z += Math.exp(v - m);
  const lz = m + Math.log(z), out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i] - lz;
  return out;
}
export function topk(logprobs, k) {
  const idx = Array.from(logprobs.keys()).sort((i, j) => logprobs[j] - logprobs[i]).slice(0, k);
  return idx.map(i => ({ id: i, logprob: logprobs[i], prob: Math.exp(logprobs[i]) }));
}

// logits for one hidden row z (length d) against the tied embedding matrix E (V x d)
function tiedLogits(E, V, d, z, zo = 0) {
  const out = new Float32Array(V);
  for (let w = 0; w < V; w++) out[w] = dot(E, w * d, z, zo, d);
  return out;
}

// ---------------------------------------------------------------- the local gated CNN
export class LocalCNN {
  constructor(t) {
    [this.V, this.d] = t["emb.weight"].shape; this.E = t["emb.weight"].data;
    this.layers = [];
    for (let l = 0; t[`convs.${l}.weight`]; l++) {
      const w = t[`convs.${l}.weight`], [out, inp, k] = w.shape;            // PyTorch layout [out][in][k]
      const Wk = [];                                                         // re-laid out as [k][out][in], contiguous over `in`
      for (let kk = 0; kk < k; kk++) {
        const m = new Float32Array(out * inp);
        for (let o = 0; o < out; o++) for (let i = 0; i < inp; i++) m[o * inp + i] = w.data[(o * inp + i) * k + kk];
        Wk.push(m);
      }
      this.layers.push({ Wk, b: t[`convs.${l}.bias`].data, out, k });
    }
    this.kernel = this.layers[0].k; this.receptiveField = 1 + (this.kernel - 1) * this.layers.length;
    this.normW = t["norm.weight"].data; this.normB = t["norm.bias"].data;
  }

  // Returns the normalised hidden states (T x d) and, per layer, the gate values sigma(g) (T x d).
  forward(ids, blank = -1) {                                              // blank: position whose embedding is replaced by zeros
    const T = ids.length, d = this.d;
    const h = new Float32Array(T * d);
    for (let t = 0; t < T; t++) if (t !== blank) h.set(this.E.subarray(ids[t] * d, ids[t] * d + d), t * d);
    const gates = [], record = [];
    for (const { Wk, b, out, k } of this.layers) {
      const a = new Float32Array(T * out), s = new Float32Array(T * d), u = new Float32Array(T * d), g = new Float32Array(T * d), hIn = h.slice();
      for (let t = 0; t < T; t++) {
        for (let o = 0; o < out; o++) {
          let acc = b[o];
          for (let kk = 0; kk < k; kk++) { const src = t - (k - 1) + kk; if (src >= 0) acc += dot(Wk[kk], o * d, h, src * d, d); }
          a[t * out + o] = acc;
        }
      }
      for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) {          // h += u * sigmoid(g), u = first d channels, g = last d
        const sg = sigmoid(a[t * out + d + i]); s[t * d + i] = sg; u[t * d + i] = a[t * out + i]; g[t * d + i] = a[t * out + d + i]; h[t * d + i] += a[t * out + i] * sg;
      }
      gates.push(s); record.push({ hIn, u, g, gate: s });                     // a = (u; g) before the sigmoid, gate = sigma(g)
    }
    const hFinal = h.slice();
    for (let t = 0; t < T; t++) {                                           // LayerNorm over d, eps 1e-5
      let mean = 0; for (let i = 0; i < d; i++) mean += h[t * d + i]; mean /= d;
      let v = 0; for (let i = 0; i < d; i++) { const x = h[t * d + i] - mean; v += x * x; } v /= d;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let i = 0; i < d; i++) h[t * d + i] = (h[t * d + i] - mean) * inv * this.normW[i] + this.normB[i];
    }
    return { h, gates, T, record, hFinal };
  }

  logitsAt(state, t) { return tiedLogits(this.E, this.V, this.d, state.h, t * this.d); }

  // "logit lens": the logits the network would produce from an intermediate stream vector, through the final LayerNorm
  logitsFromStream(row) {
    const d = this.d, z = new Float32Array(d);
    let mean = 0; for (let i = 0; i < d; i++) mean += row[i]; mean /= d;
    let v = 0; for (let i = 0; i < d; i++) { const x = row[i] - mean; v += x * x; } v /= d;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < d; i++) z[i] = (row[i] - mean) * inv * this.normW[i] + this.normB[i];
    return tiedLogits(this.E, this.V, d, z);
  }
}

// ---------------------------------------------------------------- the two-layer LSTM
export class LSTMLM {
  constructor(t) {
    [this.V, this.d] = t["emb.weight"].shape; this.E = t["emb.weight"].data;
    this.layers = [];
    for (let l = 0; t[`rnn.weight_ih_l${l}`]; l++) {
      const Wih = t[`rnn.weight_ih_l${l}`], Whh = t[`rnn.weight_hh_l${l}`];
      const bih = t[`rnn.bias_ih_l${l}`].data, bhh = t[`rnn.bias_hh_l${l}`].data;
      const b = new Float32Array(bih.length); for (let i = 0; i < b.length; i++) b[i] = bih[i] + bhh[i];
      this.layers.push({ Wih: Wih.data, Whh: Whh.data, b, inp: Wih.shape[1], hidden: Whh.shape[1] });
    }
    this.hidden = this.layers[0].hidden;
    this.projW = t["proj.weight"].data; this.projB = t["proj.bias"].data;      // (d x hidden)
  }

  // Returns the top-layer outputs (T x hidden) and, per layer, the forget gates and memory cells (T x hidden).
  // one recurrent step of layer `layer`: from the input row x (offset xo) and the previous (h, c), returns the gates and the new states
  step(layer, x, xo, h, c) {
    const { Wih, Whh, b, inp: nin, hidden: H } = this.layers[layer], z = new Float32Array(4 * H);
    for (let r = 0; r < 4 * H; r++) z[r] = b[r] + dot(Wih, r * nin, x, xo, nin) + dot(Whh, r * H, h, 0, H);
    const hn = new Float32Array(H), cn = new Float32Array(H), I = new Float32Array(H), F = new Float32Array(H), G = new Float32Array(H), O = new Float32Array(H);
    for (let j = 0; j < H; j++) {                                          // PyTorch gate order: input, forget, cell, output
      const i = sigmoid(z[j]), f = sigmoid(z[H + j]), g = Math.tanh(z[2 * H + j]), o = sigmoid(z[3 * H + j]);
      cn[j] = f * c[j] + i * g; hn[j] = o * Math.tanh(cn[j]); I[j] = i; F[j] = f; G[j] = g; O[j] = o;
    }
    return { h: hn, c: cn, I, F, G, O };
  }

  // Returns the top-layer outputs (T x hidden) and, per layer, the gates, candidates, memory cells and outputs (T x hidden).
  // blank: position whose embedding is replaced by zeros.
  forward(ids, blank = -1) {
    const T = ids.length, d = this.d;
    let inp = new Float32Array(T * d);
    for (let t = 0; t < T; t++) if (t !== blank) inp.set(this.E.subarray(ids[t] * d, ids[t] * d + d), t * d);
    const forget = [], cell = [], record = [];
    for (let l = 0; l < this.layers.length; l++) {
      const { inp: nin, hidden: H } = this.layers[l];
      const out = new Float32Array(T * H), F = new Float32Array(T * H), C = new Float32Array(T * H);
      const I = new Float32Array(T * H), G = new Float32Array(T * H), O = new Float32Array(T * H);
      let h = new Float32Array(H), c = new Float32Array(H);
      for (let t = 0; t < T; t++) {
        const r = this.step(l, inp, t * nin, h, c);
        h = r.h; c = r.c; out.set(h, t * H); F.set(r.F, t * H); C.set(c, t * H); I.set(r.I, t * H); G.set(r.G, t * H); O.set(r.O, t * H);
      }
      forget.push(F); cell.push(C); record.push({ x: inp, nin, input: I, forget: F, candidate: G, output: O, cell: C, h: out }); inp = out;
    }
    return { out: inp, forget, cell, T, record };
  }

  // Logits at position `to` when the token at `blank` is erased, resuming from the states recorded just before `blank`.
  // Only the steps from `blank` to `to` are recomputed.
  logitsWithBlank(state, ids, blank, to) {
    const d = this.d, L = this.layers.length, H = this.hidden;
    let h = [], c = [];
    for (let l = 0; l < L; l++) {
      h.push(blank > 0 ? state.record[l].h.slice((blank - 1) * H, blank * H) : new Float32Array(H));
      c.push(blank > 0 ? state.record[l].cell.slice((blank - 1) * H, blank * H) : new Float32Array(H));
    }
    let top = null;
    for (let t = blank; t <= to; t++) {
      let x = t === blank ? new Float32Array(d) : this.E.subarray(ids[t] * d, ids[t] * d + d);
      for (let l = 0; l < L; l++) { const r = this.step(l, x, 0, h[l], c[l]); h[l] = r.h; c[l] = r.c; x = r.h; }
      top = x;
    }
    const z = new Float32Array(d);
    for (let i = 0; i < d; i++) z[i] = this.projB[i] + dot(this.projW, i * H, top, 0, H);
    return tiedLogits(this.E, this.V, d, z);
  }

  // the projected vector z_t (length d) whose dot products with the embeddings are the logits
  projAt(state, t) {
    const H = this.hidden, d = this.d, z = new Float32Array(d);
    for (let i = 0; i < d; i++) z[i] = this.projB[i] + dot(this.projW, i * H, state.out, t * H, H);
    return z;
  }

  logitsAt(state, t) { return tiedLogits(this.E, this.V, this.d, this.projAt(state, t)); }
}

// ---------------------------------------------------------------- analyses shared by both models
// log-probability the model gave to the token that actually came, at every position
export function nextTokenLogprobs(model, ids) {
  const state = model.forward(ids), out = new Float32Array(ids.length - 1);
  for (let t = 0; t < ids.length - 1; t++) out[t] = logSoftmax(model.logitsAt(state, t))[ids[t + 1]];
  return out;
}

// prediction at `position` when the model only sees the last k tokens, for each k
export function contextCurve(model, ids, position, ks) {
  const rows = [];
  for (const k of ks) {
    if (k > position + 1) break;
    const ctx = ids.slice(position + 1 - k, position + 1);
    const lp = logSoftmax(model.logitsAt(model.forward(ctx), ctx.length - 1));
    const best = topk(lp, 1)[0];
    rows.push({ k, top1: best.id, top1Prob: best.prob, actualLogprob: position + 1 < ids.length ? lp[ids[position + 1]] : null });
  }
  return rows;
}

// cosine nearest neighbours of token `id` in the embedding matrix of a model
export function nearestTokens(model, id, k = 8) {
  const { E, V, d } = model, norms = model._norms || (model._norms = (() => {
    const n = new Float32Array(V); for (let w = 0; w < V; w++) n[w] = Math.sqrt(dot(E, w * d, E, w * d, d)); return n; })());
  const sims = new Float32Array(V);
  for (let w = 0; w < V; w++) sims[w] = dot(E, w * d, E, id * d, d) / (norms[w] * norms[id] + 1e-9);
  return Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).filter(w => w !== id).slice(0, k).map(w => ({ id: w, sim: sims[w] }));
}

// KL divergence, in bits, between the full prediction and the prediction with one past token erased,
// for the last `n` tokens up to `position` (position included). Returns [{t, bits}].
export function klBits(lpFull, lpOther) {
  let kl = 0;
  for (let w = 0; w < lpFull.length; w++) { const p = Math.exp(lpFull[w]); if (p > 1e-12) kl += p * (lpFull[w] - lpOther[w]); }
  return Math.max(0, kl) / Math.LN2;
}
export function influenceCNN(model, ids, position, lpFull) {
  const start = Math.max(0, position - model.receptiveField + 1), win = ids.slice(start, position + 1), out = [];
  for (let j = 0; j < win.length; j++) {
    const lp = logSoftmax(model.logitsAt(model.forward(win, j), win.length - 1));
    out.push({ t: start + j, bits: klBits(lpFull, lp) });
  }
  return out;
}
export function influenceLSTM(model, state, ids, position, lpFull, n = 16) {
  const out = [];
  for (let j = Math.max(0, position - n + 1); j <= position; j++) out.push({ t: j, bits: klBits(lpFull, logSoftmax(model.logitsWithBlank(state, ids, j, position))) });
  return out;
}
