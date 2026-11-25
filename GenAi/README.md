# PS 10 — Headline & Title Generator

GenAI (LLM) Mini Project · Problem Statement 10 · Roll No. 10 · LY CORE 6

Writers struggle with catchy titles. This app takes a pasted article (or just
a topic) and returns **10 headline options, each rated 1–10 for clarity**,
shown best-first in a sorted table.

## Stack

- Python 3.10+
- Streamlit (UI)
- Google Gemini 3.8 Flash via API (free tier, `google-genai` SDK)

No model training — the LLM does the generation; Python sorts, displays and
exports the results.

## How to run

### Option A — Gemini API (free key)

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and paste your key:
   `GEMINI_API_KEY=your_key_here`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `streamlit run app.py`
5. Pick **Gemini (API key)** in the app, paste an article or a topic,
   click **Generate headlines**.

### Option B — local model with Ollama (no key, works offline)

1. Install Ollama from https://ollama.com/download and open it
   (it runs in the menu bar).
2. Pull a small model once (about 2 GB download):
   `ollama pull llama3.2`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `streamlit run app.py`
5. Pick **Ollama (local model)** in the app, keep the model name
   `llama3.2`, paste an article or a topic, click **Generate headlines**.

Local generation is slower than the API (a few seconds per run on an
M-series Mac) but needs no key and no internet. `llama3.1:8b` also works
if you want a bigger model — pull it with `ollama pull llama3.1:8b`.

### Option C — OpenRouter free model (no local install)

1. Get a free key at https://openrouter.ai/keys (sign up, then create a key).
2. Copy `.env.example` to `.env` and paste your key:
   `OPENROUTER_API_KEY=sk-or-v1-...`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `streamlit run app.py`
5. Pick **OpenRouter (free model)** in the app, keep the default model
   (`google/gemma-4-31b-it:free`), paste an article or a
   topic, click **Generate headlines**. If one free model fails, the app
   tries the next one automatically. Never commit your `.env` file.

## How it works

1. The article/topic is inserted into a fixed prompt that asks Gemini for
   exactly 10 headlines as a JSON array, each with a `clarity_score` (1–10),
   with rules: vary the style, under 15 words each, no invented facts.
2. `parse_headlines()` strips any markdown fence, parses the JSON, clamps
   scores to 1–10, drops bad rows, and sorts best-first.
3. The UI shows the top headline, a ranked table, and a CSV download.

## Files

- `app.py` — the whole app (prompt, parsing, Streamlit UI)
- `requirements.txt` — dependencies
- `.env.example` — copy to `.env` and add your key (never commit `.env`)
- `.gitignore` — keeps `.env` and `venv/` out of Git

## Notes

- Clarity scores are the model's own ratings, not measured readership data.
- If the model returns bad JSON, the app shows the error instead of fake
  headlines — fix by re-running or rewording the input.
