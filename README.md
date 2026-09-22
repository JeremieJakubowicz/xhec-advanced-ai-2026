# Advanced AI: Language Models & Agents — X-HEC, Fall 2026

Public material for the course. Everything runs on the free tier of Google Colab; no paid service and no account other than Google are needed.

## Sessions

| #   | session                | topic                                                                         | GPU            | open in Colab                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --- | ---------------------- | ----------------------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Session 1](Session1/) | Shannon, Markov models, the memorization trap, tokenization                   | no             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session1/Session1a-Shannon-Markov.ipynb) part A · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session1/Session1b-Tokenization.ipynb) part B           |
| 2   | [Session 2](Session2/) | From the lookup table to neural networks: CNN, RNN, LSTM, vanishing gradients | T4 recommended | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session2/Session2a-From-Table-To-Network.ipynb) part A · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JeremieJakubowicz/xhec-advanced-ai-2026/blob/main/Session2/Session2b-Going-Non-Local.ipynb) part B |

Each session folder has its own README explaining how to run the notebooks, on Colab or locally, and what to do if something goes wrong.

## Data

`data/` holds the datasets the notebooks download by themselves: `hugotexts.txt`, works of Victor Hugo (14 MB, public domain, accents partly stripped by the digitisation), and `hugo_bpe_8k.json`, an 8,000-token tokenizer built with the recipe of session 1 on the training split of the corpus. Pre-trained model checkpoints live in each session's `checkpoints/` folder.

## Instructors

[Jérémie Jakubowicz](https://www.linkedin.com/in/j%C3%A9r%C3%A9mie-jakubowicz-8b1277133/) and [Nicolas Schuhl](https://www.linkedin.com/in/nicolasschuhl/).
