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
  forward(ids) {
    const T = ids.length, d = this.d;
    const h = new Float32Array(T * d);
    for (let t = 0; t < T; t++) h.set(this.E.subarray(ids[t] * d, ids[t] * d + d), t * d);
    const gates = [];
    for (const { Wk, b, out, k } of this.layers) {
      const a = new Float32Array(T * out), s = new Float32Array(T * d);
      for (let t = 0; t < T; t++) {
        for (let o = 0; o < out; o++) {
          let acc = b[o];
          for (let kk = 0; kk < k; kk++) { const src = t - (k - 1) + kk; if (src >= 0) acc += dot(Wk[kk], o * d, h, src * d, d); }
          a[t * out + o] = acc;
        }
      }
      for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) {          // h += u * sigmoid(g), u = first d channels, g = last d
        const g = sigmoid(a[t * out + d + i]); s[t * d + i] = g; h[t * d + i] += a[t * out + i] * g;
      }
      gates.push(s);
    }
    for (let t = 0; t < T; t++) {                                           // LayerNorm over d, eps 1e-5
      let mean = 0; for (let i = 0; i < d; i++) mean += h[t * d + i]; mean /= d;
      let v = 0; for (let i = 0; i < d; i++) { const x = h[t * d + i] - mean; v += x * x; } v /= d;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let i = 0; i < d; i++) h[t * d + i] = (h[t * d + i] - mean) * inv * this.normW[i] + this.normB[i];
    }
    return { h, gates, T };
  }

  logitsAt(state, t) { return tiedLogits(this.E, this.V, this.d, state.h, t * this.d); }
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
  forward(ids) {
    const T = ids.length, d = this.d;
    let inp = new Float32Array(T * d);
    for (let t = 0; t < T; t++) inp.set(this.E.subarray(ids[t] * d, ids[t] * d + d), t * d);
    const forget = [], cell = [];
    for (const { Wih, Whh, b, inp: nin, hidden: H } of this.layers) {
      const out = new Float32Array(T * H), F = new Float32Array(T * H), C = new Float32Array(T * H);
      let h = new Float32Array(H), c = new Float32Array(H);
      const z = new Float32Array(4 * H);
      for (let t = 0; t < T; t++) {
        for (let r = 0; r < 4 * H; r++) z[r] = b[r] + dot(Wih, r * nin, inp, t * nin, nin) + dot(Whh, r * H, h, 0, H);
        const hn = new Float32Array(H), cn = new Float32Array(H);
        for (let j = 0; j < H; j++) {                                        // PyTorch gate order: input, forget, cell, output
          const i = sigmoid(z[j]), f = sigmoid(z[H + j]), g = Math.tanh(z[2 * H + j]), o = sigmoid(z[3 * H + j]);
          cn[j] = f * c[j] + i * g; hn[j] = o * Math.tanh(cn[j]); F[t * H + j] = f; C[t * H + j] = cn[j];
        }
        h = hn; c = cn; out.set(h, t * H);
      }
      forget.push(F); cell.push(C); inp = out;
    }
    return { out: inp, forget, cell, T };
  }

  logitsAt(state, t) {
    const H = this.hidden, d = this.d, z = new Float32Array(d);
    for (let i = 0; i < d; i++) z[i] = this.projB[i] + dot(this.projW, i * H, state.out, t * H, H);
    return tiedLogits(this.E, this.V, this.d, z);
  }
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
