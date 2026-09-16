# Advanced AI: Language Models & Agents — X-HEC, Fall 2026

Public material for the course. Notebooks open directly in Google Colab and download what they need from this repository; no account and no paid service is required.

## Data

| file | content | size | source |
|---|---|---|---|
| `data/hugotexts.txt` | Works of Victor Hugo (speeches, plays, poems, novels), plain UTF-8 text, accents partly stripped by the digitisation | 14 MB | public domain texts |

Load it from a notebook with:

```python
import os, urllib.request
URL = "https://raw.githubusercontent.com/JeremieJakubowicz/xhec-advanced-ai-2026/main/data/hugotexts.txt"
if not os.path.exists("hugotexts.txt"):
    urllib.request.urlretrieve(URL, "hugotexts.txt")
hugo = open("hugotexts.txt", encoding="utf-8").read()
```

## Instructors

Jérémie Jakubowicz (theory) and Nicolas Schuhl (practice).
