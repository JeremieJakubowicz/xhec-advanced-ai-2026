# Session 2 — From the lookup table to recurrent networks

Two notebooks. They train small language models live, so a GPU runtime is recommended (free on Colab). No account other than Google.

| notebook                                | what it does                                                                                                                                                                                                                                   | run time on Colab (T4 GPU) | open                                                                                                                                                                                                                    |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Session2a-From-Table-To-Network.ipynb` | Self-supervision, one measurement for every language model (surprise, perplexity, bits per character), the smoothed table on the scoreboard, a convolutional network trained live and compared to it, embeddings, how far back the model looks | about 6 minutes            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session2/Session2a-From-Table-To-Network.ipynb) |
| `Session2b-Going-Non-Local.ipynb`       | Recurrent networks: the simple RNN and why it loses, vanishing gradients measured, the LSTM and why it wins, the limits that lead to attention                                                                                                 | about 8 minutes            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session2/Session2b-Going-Non-Local.ipynb)       |

Do part A before part B: part B reloads everything part A built and starts where it ends. Both notebooks use the corpus of session 1 and a tokenizer built with the same recipe as in session 1, retrained on the training split only.

## Run on Google Colab (recommended)

1. Click the badge. Colab opens a copy of the notebook taken from this repository.
2. To keep your changes, **File > Save a copy in Drive** right away.
3. Choose a GPU: **Runtime > Change runtime type > T4 GPU**. On CPU the notebooks still run, but the two-minute live trainings only get a few hundred steps and the loss curves are less telling.
4. Run the first code cell, which installs the one package Colab does not ship. If Colab asks to restart the session, accept, then **Runtime > Run all**.
5. The notebooks download what they need from this repository: the corpus (14 MB), the tokenizer (0.5 MB) and the pre-trained checkpoints (7 to 12 MB each). No Hugging Face account, nothing to upload.

Each live training cell runs for exactly two minutes, whatever the machine. The checkpoints loaded afterwards were trained for fifteen minutes with the same code and the same data; they are what the scoreboard is built on, so that everyone gets the same numbers.

## Run locally

Python 3.10 or later, PyTorch 2.x. A recent laptop GPU (Apple M-series, or any NVIDIA card) is enough; on a plain CPU count about three times the Colab durations.

```
pip install torch tokenizers pandas matplotlib scikit-learn jupyterlab
jupyter lab
```

Open the notebooks from this folder. Downloaded files are cached next to the notebooks.

## If something goes wrong

- **Colab offers no GPU right now.** Run on CPU: everything works, the live trainings just learn less in their two minutes. The checkpoint-based results are unaffected.
- **`load_checkpoint` fails with a size mismatch.** The model class in the notebook was edited: the checkpoints expect the architectures exactly as written (256-dimensional embeddings, four convolutional layers, two recurrent layers of 512 units).
- **A download hangs.** Run the cell again; the files are small. As a last resort, download them by hand from `data/` and `Session2/checkpoints/` in this repository and upload them next to the notebook with the *Files* pane.
- **Out of memory on the GPU.** Reduce `batch_size` in `train` from 64 to 32.

## What to bring

The notions of part A of session 1 (context, next token, memory) and of part B (tokens). Gradient descent and the idea of a loss function, from any introduction to machine learning.
