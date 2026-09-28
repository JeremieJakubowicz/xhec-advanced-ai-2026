# Session 3 — Attention, and the Transformer

One notebook. It trains a small Transformer live and compares it with the networks of session 2, so a GPU runtime is recommended (free on Colab). No account other than Google.

| notebook                   | what it does                                                                                                                                                                                                                                                                                                                                                                                                   | run time on Colab (T4 GPU) | open                                                                                                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Session3-Attention.ipynb` | The three limits of the LSTM; attention as a soft lookup over the past (queries, keys, values, causal mask); several heads and position vectors; the Transformer block and the whole model, written by hand; a two-minute live training and the fifteen-minute checkpoint; what the heads look at; what attention buys, measured: context curves, gradient reach, speed, a repeated-token test, head ablations | about 12 minutes           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session3/Session3-Attention.ipynb) |

The notebook starts where session 2 ended: it reloads the three checkpoints of session 2 (CNN, simple RNN, LSTM) and scores them again with the same corpus, tokenizer and measurement, so that the Transformer joins the same scoreboard.

## Run on Google Colab (recommended)

1. Click the badge. Colab opens a copy of the notebook taken from this repository.
2. To keep your changes, **File > Save a copy in Drive** right away.
3. Choose a GPU: **Runtime > Change runtime type > T4 GPU**. On CPU the notebook still runs, but the two-minute live training only gets a few hundred steps, and the measurements of section 8 take several minutes each.
4. Run the first code cell, which installs the one package Colab does not ship. If Colab asks to restart the session, accept, then **Runtime > Run all**.
5. The notebook downloads what it needs from this repository: the corpus (14 MB), the tokenizer (0.5 MB) and four checkpoints (6 to 12 MB each). No Hugging Face account, nothing to upload.

The live training cell runs for exactly two minutes, whatever the machine. The checkpoint loaded afterwards was trained for fifteen minutes with the same code and the same data, like those of session 2; the scoreboard is built on the checkpoints, so that everyone gets the same numbers.

## Run locally

Python 3.10 or later, PyTorch 2.x. A recent laptop GPU (Apple M-series, or any NVIDIA card) is enough; on a plain CPU count about three times the Colab durations.

```
pip install torch tokenizers pandas matplotlib scikit-learn jupyterlab
jupyter lab
```

Open the notebook from this folder. Downloaded files are cached next to it.

## If something goes wrong

- **Colab offers no GPU right now.** Run on CPU: everything works, the live training just learns less in its two minutes. The checkpoint-based results are unaffected.
- **`load_checkpoint` fails with a size mismatch.** A model class in the notebook was edited: the checkpoints expect the architectures exactly as written (256-dimensional embeddings, four blocks of four heads, a context of 128 tokens for the Transformer; the session 2 architectures for the others).
- **A download hangs.** Run the cell again; the files are small. As a last resort, download them by hand from `data/`, `Session2/checkpoints/` and `Session3/checkpoints/` in this repository and upload them next to the notebook with the *Files* pane.
- **Out of memory on the GPU.** Reduce `batch_size` in `train` from 64 to 32, and `B` in `tokens_per_second` from 32 to 8.

## What to bring

Session 2: the residual stream of the CNN, the LSTM and its three limits, the context curve and the single measurement in bits per character. Matrix products and the softmax, from any introduction to machine learning.
