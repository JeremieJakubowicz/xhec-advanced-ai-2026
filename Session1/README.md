# Session 1 — Shannon, Markov models, the memorization trap, and tokenization

Two notebooks. No GPU needed, no account other than Google.

| notebook                         | what it does                                                                                                                                                    | run time on Colab (CPU) | open                                                                                                                                                                                                                              |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Session1a-Shannon-Markov.ipynb` | Shannon's experiment, Markov language models on characters and words, and what happens when the memory grows: the model recites the corpus and becomes mute     | about 4 minutes         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session1/Session1a-Shannon-Markov.ipynb) |
| `Session1b-Tokenization.ipynb`   | Byte-Pair Encoding from scratch, six real tokenizers compared on languages, code and numbers, and the special tokens that encode dialogue, tool calls, thinking | about 1 minute          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session1/Session1b-Tokenization.ipynb)   |

Do part A before part B: part B starts where part A ends.

## Run on Google Colab (recommended)

1. Click the badge. Colab opens a copy of the notebook taken from this repository.
2. To keep your changes, use **File > Save a copy in Drive** right away; otherwise edits are lost when the tab closes.
3. Leave the runtime on CPU (**Runtime > Change runtime type > CPU**). A GPU brings nothing here.
4. Run the first code cell, which installs the two or three packages Colab does not ship. If Colab asks to restart the session, accept, then **Runtime > Run all**.
5. The next cell downloads the Victor Hugo corpus (14 MB) from this repository. Part B also downloads the Qwen3 tokenizer from Hugging Face (about 10 MB); no Hugging Face account is needed.

Part A spends a few minutes on the memory sweeps of section 4; the outputs are already saved in the notebook, so you can read ahead while it runs.

## Run locally

Python 3.10 or later.

```
pip install tiktoken transformers tokenizers "mistral-common[sentencepiece]" pandas matplotlib jupyterlab
jupyter lab
```

Open the notebooks from this folder. The corpus is downloaded once to `hugotexts.txt` next to the notebooks. Part A needs about 2 GB of free memory for the largest Markov models.

## If something goes wrong

- **The corpus download hangs or fails.** Run the cell again. If it keeps failing, download `data/hugotexts.txt` from this repository by hand and upload it to Colab with the *Files* pane on the left, under the name `hugotexts.txt`.
- **Out of memory in part A** (unlikely on Colab, possible on a laptop with 8 GB). Replace `hugo[:2_000_000]` by `hugo[:1_000_000]` where the training slice is defined, and drop the 4M point from the corpus-size sweep in section 4.3.
- **Hugging Face cannot be reached in part B.** Only the Qwen3 tokenizer comes from there. Remove the `hf_tok(qwen, ...)` line from the tokenizer zoo and the cells that use `qwen` in section 4; everything else runs offline.
- **A cell shows `%pip` errors.** You are probably running the notebook outside Jupyter; install the packages from the command line instead and skip that cell.

## What to bring

Python basics, the idea of a probability distribution. No neural network appears in this session.
