"""PS 10 — Headline & Title Generator.

Paste an article (or just a topic) and get 10 headline options, each rated
1-10 for clarity, shown best-first in a sorted table.

LLM: Google Gemini 3.8 Flash via API (free key from Google AI Studio).
Key is read from the GEMINI_API_KEY environment variable or a .env file —
never hard-code it. See README.md.
"""

import json
import os
import re

import requests
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None

MODEL_NAME = "gemini-3.8-flash"
# If the first model is overloaded (503) or unavailable, try the next one.
MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-flash-latest"]

OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free models on OpenRouter — tried in order until one answers.
# Verified live 2026-11-23 against the /api/v1/models list; the free
# list rotates, so keep this current if a model starts 404ing.
OPENROUTER_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
]

PROMPT_TEMPLATE = """You are a headline editor. Given the article text or topic below,
write exactly 10 different headline options for it.

Rules:
- Vary the style: some informative, some curiosity-driven, some short and punchy.
- Keep each headline under 15 words.
- Do not invent facts that are not in the article/topic.
- Rate each headline for CLARITY from 1 to 10 (10 = instantly clear what the
  article is about, 1 = vague or confusing).

Return ONLY a JSON array, no other text, no markdown fences. Each item must
have exactly these keys:
{{"headline": "...", "clarity_score": 8}}

Article / topic:
---
{article}
---
"""


def parse_headlines(raw_text):
    """Extract the JSON array of headlines from the model reply.

    The model is asked for raw JSON, but it may still wrap it in a markdown
    code fence, so strip fences and take the first [...] block.
    Returns a list of dicts with keys 'headline' and 'clarity_score'.
    Raises ValueError on bad JSON or wrong shape.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in the model reply.")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("Model reply was not a JSON list.")
    headlines = []
    for item in data:
        if not isinstance(item, dict) or "headline" not in item:
            continue
        try:
            score = int(item.get("clarity_score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(1, min(10, score))
        headlines.append(
            {"headline": str(item["headline"]).strip(), "clarity_score": score}
        )
    if not headlines:
        raise ValueError("Model reply contained no usable headlines.")
    headlines.sort(key=lambda h: h["clarity_score"], reverse=True)
    return headlines


def generate_headlines(article, api_key):
    """Call Gemini and return (headlines, model_used).

    Tries each model in MODEL_CANDIDATES in order until one answers, so a
    503 overload on one model falls through to the next instead of failing.
    Raises RuntimeError if every model fails.
    """
    if genai is None:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install -r requirements.txt"
        )
    client = genai.Client(api_key=api_key)
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name, contents=PROMPT_TEMPLATE.format(article=article)
            )
            return parse_headlines(response.text), model_name
        except Exception as exc:  # noqa: BLE001 — try the next model
            last_error = exc
    raise RuntimeError(f"All models failed. Last error: {last_error}")


def generate_headlines_ollama(article, model):
    """Call a local Ollama model and return (headlines, label).

    No API key needed — Ollama must be running on this machine
    (`ollama serve`) with the model already pulled (`ollama pull <model>`).
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(article=article),
                    }
                ],
                "stream": False,
                "format": "json",
            },
            timeout=180,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach Ollama at localhost:11434. Is Ollama running? "
            "Start it with `ollama serve` and pull a model with "
            "`ollama pull llama3.2`."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama has no model named '{model}'. "
            f"Pull it first: `ollama pull {model}`."
        )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return parse_headlines(content), f"Ollama {model} (local)"


def generate_headlines_openrouter(article, api_key, model):
    """Call a free model on OpenRouter and return (headlines, label).

    If the chosen model fails (not found, rate-limited), tries the next
    model in OPENROUTER_MODELS. Raises RuntimeError if all fail.
    """
    last_error = None
    candidates = [model] + [m for m in OPENROUTER_MODELS if m != model]
    for model_name in candidates:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(article=article),
                        }
                    ],
                },
                timeout=120,
            )
            if resp.status_code == 401:
                raise RuntimeError(
                    "OpenRouter rejected the API key (401). Check the key "
                    "in your .env file."
                )
            if resp.status_code == 402:
                raise RuntimeError(
                    "OpenRouter says this needs credits (402). "
                    "Use a :free model."
                )
            if resp.status_code in (404, 429):
                last_error = f"{resp.status_code} for {model_name}"
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return parse_headlines(content), f"OpenRouter {model_name}"
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 — try the next model
            last_error = exc
    raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")


st.set_page_config(page_title="PS 10 — Headline Generator", page_icon="📰")
st.title("📰 Headline & Title Generator")
st.caption(
    "PS 10 · GenAI (LLM) Mini Project · Paste an article or a topic, get 10 "
    "headline options rated for clarity, best first."
)

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
provider = st.selectbox(
    "LLM provider",
    ["Gemini (API key)", "Ollama (local model)", "OpenRouter (free model)"],
)
ollama_model = DEFAULT_OLLAMA_MODEL
openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
openrouter_model = OPENROUTER_MODELS[0]
if provider == "Ollama (local model)":
    ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
    st.caption(
        "Runs on your own machine — no API key, no internet needed once the "
        "model is downloaded. Install Ollama from ollama.com, then run "
        "`ollama pull llama3.2` once. Make sure Ollama is running."
    )
elif provider == "OpenRouter (free model)":
    openrouter_model = st.text_input(
        "OpenRouter model", value=OPENROUTER_MODELS[0]
    )
    if not openrouter_key:
        openrouter_key = st.text_input(
            "OpenRouter API key", type="password"
        ).strip()
        st.caption(
            "Get a free key at https://openrouter.ai/keys — or set "
            "OPENROUTER_API_KEY in your .env file. Free models end in :free."
        )
elif not api_key:
    st.warning(
        "No Gemini API key found. Get a free key at "
        "https://aistudio.google.com/apikey and set it as GEMINI_API_KEY "
        "(see README.md), then restart the app — or switch to Ollama above."
    )

article = st.text_area(
    "Article text or topic",
    height=180,
    placeholder=(
        "Paste a full article here, or just a topic — e.g. "
        "'MIT ADT wins the national robotics championship'."
    ),
)
col_a, col_b = st.columns([1, 1])
with col_a:
    tone = st.selectbox(
        "Headline style",
        ["Mixed (all styles)", "Informative", "Curiosity-driven", "Short & punchy"],
    )
with col_b:
    num = st.slider("Headlines to generate", 5, 10, 10)

needs_key = (provider == "Gemini (API key)" and not api_key) or (
    provider == "OpenRouter (free model)" and not openrouter_key
)
if st.button("Generate headlines", type="primary", disabled=needs_key):
    if not article.strip():
        st.error("Paste an article or a topic first.")
    else:
        prompt_article = article
        if tone != "Mixed (all styles)":
            prompt_article = f"[Preferred style: {tone}]\n{article}"
        with st.spinner("Generating headlines…"):
            try:
                if provider == "Ollama (local model)":
                    headlines, model_used = generate_headlines_ollama(
                        prompt_article, ollama_model.strip() or DEFAULT_OLLAMA_MODEL
                    )
                elif provider == "OpenRouter (free model)":
                    headlines, model_used = generate_headlines_openrouter(
                        prompt_article,
                        openrouter_key,
                        openrouter_model.strip() or OPENROUTER_MODELS[0],
                    )
                else:
                    headlines, model_used = generate_headlines(prompt_article, api_key)
            except Exception as exc:  # noqa: BLE001 — show the error to the user
                st.error(f"Generation failed: {exc}")
                headlines = []
        if headlines:
            headlines = headlines[:num]
            st.success(f"Generated {len(headlines)} headlines — sorted by clarity.")
            st.caption(f"Answered by model: {model_used}")
            best = headlines[0]
            st.metric(
                "Top headline (clarity %d/10)" % best["clarity_score"],
                best["headline"],
            )
            table = [
                {
                    "Rank": i + 1,
                    "Headline": h["headline"],
                    "Clarity (1–10)": h["clarity_score"],
                }
                for i, h in enumerate(headlines)
            ]
            st.dataframe(table, use_container_width=True)
            csv_lines = ["rank,headline,clarity_score"] + [
                f'{i + 1},"{h["headline"].replace(chr(34), chr(34) * 2)}",{h["clarity_score"]}'
                for i, h in enumerate(headlines)
            ]
            st.download_button(
                "Download as CSV",
                data="\n".join(csv_lines),
                file_name="headlines.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Built with Python + Streamlit. Works with Gemini, OpenRouter free "
    "models, or a local Ollama model. Headlines are generated by the LLM — "
    "clarity scores are the model's own ratings, not measured readership data."
)
