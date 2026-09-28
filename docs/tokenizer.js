// Byte-level BPE tokenizer (the Hugging Face `tokenizers` ByteLevel + BPE recipe, as used by GPT-2),
// reimplemented in plain JavaScript so that the companion site needs no library.
// Works in the browser and in Node (ES module).

const GPT2_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// The byte <-> unicode table of GPT-2: printable bytes map to themselves, the others to U+0100 and up.
function byteUnicodeTables() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const byteToChar = new Array(256), charToByte = new Map();
  bs.forEach((b, i) => { byteToChar[b] = String.fromCodePoint(cs[i]); charToByte.set(byteToChar[b], b); });
  return { byteToChar, charToByte };
}

export class ByteLevelBPE {
  constructor(tokenizerJson) {
    const model = tokenizerJson.model;
    this.vocab = new Map(Object.entries(model.vocab));                     // token string -> id
    this.idToToken = new Array(this.vocab.size);
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;
    this.ranks = new Map();                                                  // "a b" -> merge rank
    model.merges.forEach((m, i) => { const [a, b] = Array.isArray(m) ? m : m.split(" "); this.ranks.set(a + "\u0000" + b, i); });
    this.special = new Map((tokenizerJson.added_tokens || []).map(t => [t.content, t.id]));
    const { byteToChar, charToByte } = byteUnicodeTables();
    this.byteToChar = byteToChar; this.charToByte = charToByte;
    this.encoder = new TextEncoder(); this.decoder = new TextDecoder("utf-8");
    this.cache = new Map();
  }

  // BPE on one pre-token, already mapped to byte-level characters.
  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let symbols = Array.from(word);
    while (symbols.length > 1) {
      let best = null, bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = this.ranks.get(symbols[i] + "\u0000" + symbols[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) break;
      const merged = symbols[best] + symbols[best + 1];
      const next = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === symbols[best] && symbols[i + 1] === symbols[best + 1]) { next.push(merged); i++; }
        else next.push(symbols[i]);
      }
      symbols = next;
    }
    this.cache.set(word, symbols);
    return symbols;
  }

  // text -> [{id, token, text}] : id in the vocabulary, token as stored (byte-level chars), text as decoded bytes
  tokenize(text) {
    const out = [];
    for (const piece of text.match(GPT2_PATTERN) || []) {
      const mapped = Array.from(this.encoder.encode(piece), b => this.byteToChar[b]).join("");
      for (const sym of this.bpe(mapped)) {
        const id = this.vocab.get(sym);
        if (id === undefined) throw new Error(`token not in vocabulary: ${JSON.stringify(sym)}`);
        out.push({ id, token: sym, text: this.decodeToken(sym) });
      }
    }
    return out;
  }

  encode(text) { return this.tokenize(text).map(t => t.id); }

  // the pre-tokenizer's pieces of a text (the regex chunks), in order
  pieces(text) { return text.match(GPT2_PATTERN) || []; }

  // how one piece becomes tokens: its UTF-8 bytes, their byte-level characters, then every merge in order
  explain(piece) {
    const bytes = Array.from(this.encoder.encode(piece));
    const mapped = bytes.map(b => this.byteToChar[b]);
    let symbols = mapped.slice(); const steps = [{ symbols: symbols.slice(), rank: null, merged: null }];
    while (symbols.length > 1) {
      let best = null, bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = this.ranks.get(symbols[i] + "\u0000" + symbols[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) break;
      const merged = symbols[best] + symbols[best + 1], next = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === symbols[best] && symbols[i + 1] === symbols[best + 1]) { next.push(merged); i++; }
        else next.push(symbols[i]);
      }
      symbols = next; steps.push({ symbols: symbols.slice(), rank: bestRank, merged });
    }
    return { piece, bytes, mapped, steps, ids: symbols.map(sym => this.vocab.get(sym)) };
  }

  decodeToken(sym) {
    const bytes = new Uint8Array(Array.from(sym, ch => this.charToByte.get(ch)));
    return this.decoder.decode(bytes);
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) for (const ch of this.idToToken[id]) bytes.push(this.charToByte.get(ch));
    return this.decoder.decode(new Uint8Array(bytes));
  }

  get vocabSize() { return this.idToToken.length; }
}
