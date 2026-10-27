from __future__ import annotations

import pickle
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    _keras_models = import_module("tensorflow.keras.models")
    _keras_utils = import_module("tensorflow.keras.utils")
except ModuleNotFoundError:
    _keras_models = import_module("keras.models")
    _keras_utils = import_module("keras.utils")

load_model = _keras_models.load_model
pad_sequences = _keras_utils.pad_sequences


APP_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = APP_DIR / "tokenizer.pkl"
MODEL_PATH = APP_DIR / "lstm_model.h5"
MAX_LEN_PATH = APP_DIR / "max_len.pkl"

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #151d31 100%);
        color: #f3f4f6;
      }
      [data-testid="stSidebar"] {
        background: rgba(13, 20, 38, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
      }
      .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(148, 163, 184, 0.23);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(30, 41, 59, .92), rgba(30, 41, 59, .45));
        margin-bottom: 1.2rem;
      }
      .eyebrow {
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        font-weight: 700;
        color: #a5b4fc;
      }
      .hero h1 { margin: .35rem 0 .4rem 0; font-size: 2.25rem; }
      .hero p { color: #cbd5e1; margin-bottom: 0; }
      .word-card {
        padding: 1.3rem 1.4rem;
        border-radius: 18px;
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        min-height: 145px;
      }
      .word-label { color: #cbd5e1; font-size: .9rem; }
      .word-value { color: #a5b4fc; font-size: 2rem; font-weight: 800; margin: .25rem 0; overflow-wrap: anywhere; }
      .muted { color: #94a3b8; font-size: .9rem; }
      div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, .60);
        border: 1px solid rgba(148, 163, 184, .2);
        padding: .85rem 1rem;
        border-radius: 14px;
      }
      .stButton > button, .stFormSubmitButton > button {
        border-radius: 11px;
        border: 0;
        font-weight: 700;
        min-height: 2.8rem;
      }
      hr { border-color: rgba(148, 163, 184, .18); }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_max_len(raw_value: Any) -> int:
    """Read a max sequence length saved as an integer or in a small container."""
    if isinstance(raw_value, (int, np.integer)):
        value = int(raw_value)
        if value > 0:
            return value

    if isinstance(raw_value, dict):
        accepted_keys = (
            "max_len",
            "max_sequence_len",
            "max_sequence_length",
            "sequence_length",
            "input_length",
        )
        for key in accepted_keys:
            candidate = raw_value.get(key)
            if isinstance(candidate, (int, np.integer)) and int(candidate) > 0:
                return int(candidate)

    if isinstance(raw_value, (list, tuple, np.ndarray)) and len(raw_value) == 1:
        candidate = raw_value[0]
        if isinstance(candidate, (int, np.integer)) and int(candidate) > 0:
            return int(candidate)

    raise ValueError(
        "Could not read a positive sequence length from max_len.pkl. "
        "Save an integer such as max_sequence_len in that file."
    )


def model_sequence_length(model: Any, saved_max_len: int) -> int:
    """Prefer the model's expected input width; fall back to the saved max length."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0] if shape else None

    if shape is not None and len(shape) >= 2 and shape[-1] is not None:
        length = int(shape[-1])
        if length > 0:
            return length

    # In many next-word projects, max_len is the full sequence length and
    # the model is trained on sequences padded to max_len - 1.
    return max(1, saved_max_len - 1)


@st.cache_resource(show_spinner="Loading your trained LSTM model...")
def load_artifacts(
    tokenizer_path: str,
    model_path: str,
    max_len_path: str,
    _tokenizer_mtime: float,
    _model_mtime: float,
    _max_len_mtime: float,
):
    # Keep artifact timestamps as cache-key inputs so changed files reload.
    _ = (_tokenizer_mtime, _model_mtime, _max_len_mtime)

    # Pickle files can execute code while being loaded. Only use files you trust.
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    with open(max_len_path, "rb") as file:
        saved_max_len = read_max_len(pickle.load(file))

    # compile=False avoids requiring the original training loss/optimizer objects.
    model = load_model(model_path, compile=False)
    sequence_length = model_sequence_length(model, saved_max_len)
    word_index = getattr(tokenizer, "word_index", None)
    if not isinstance(word_index, dict) or not word_index:
        raise ValueError("tokenizer.pkl does not appear to contain a fitted Keras Tokenizer.")

    index_word = getattr(tokenizer, "index_word", None)
    if not isinstance(index_word, dict) or not index_word:
        index_word = {int(index): word for word, index in word_index.items()}

    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, list):
        output_shape = output_shape[0] if output_shape else None
    if output_shape is not None and len(output_shape) > 1 and output_shape[-1] is not None:
        output_vocab_size = int(output_shape[-1])
        if output_vocab_size < 2:
            raise ValueError("The model output has fewer than two classes; check lstm_model.h5.")

    return tokenizer, model, saved_max_len, sequence_length, index_word


def predict_next_words(
    text: str,
    tokenizer: Any,
    model: Any,
    sequence_length: int,
    index_word: dict,
    top_k: int,
) -> tuple[list[dict[str, float | str]], int]:
    sequences = tokenizer.texts_to_sequences([text])
    sequence = sequences[0] if sequences else []
    if not sequence:
        raise ValueError(
            "I couldn't recognize any words from this text. Try words that appear in the model's training vocabulary."
        )

    padded = pad_sequences(
        [sequence],
        maxlen=sequence_length,
        padding="pre",
        truncating="pre",
    )
    scores = np.asarray(model.predict(padded, verbose=0)).reshape(-1).astype(float)

    # Usually the model returns softmax probabilities. Convert logits to a
    # probability distribution if the output is not already non-negative.
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned non-finite prediction values.")
    if np.all(scores >= 0) and scores.sum() > 0 and scores.max() <= 1.0001:
        probabilities = scores / scores.sum()
    else:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / exp_scores.sum()

    candidates: list[dict[str, float | str]] = []
    # Sort all classes so missing/padding token IDs can be skipped cleanly.
    for index in np.argsort(probabilities)[::-1]:
        word = index_word.get(int(index))
        if not word or int(index) == 0:
            continue
        candidates.append(
            {"Word": str(word), "Probability": float(probabilities[index])}
        )
        if len(candidates) >= top_k:
            break

    if not candidates:
        raise ValueError("No valid next-word candidates were found in the tokenizer vocabulary.")
    return candidates, len(sequence)


with st.sidebar:
    st.markdown("## 🧠 Model setup")
    st.caption("The app loads these three files from the same folder as `app.py`.")
    required_files = [
        ("Tokenizer", TOKENIZER_PATH),
        ("Trained model", MODEL_PATH),
        ("Maximum length", MAX_LEN_PATH),
    ]
    for label, path in required_files:
        if path.is_file():
            st.success(f"{label}: `{path.name}` found")
        else:
            st.error(f"{label}: `{path.name}` missing")
    st.divider()
    st.markdown("### How it works")
    st.write("1. Enter a starting phrase.\n2. The app converts it into token IDs.\n3. The LSTM ranks possible next words.")
    st.caption("Tip: use a phrase similar to the text used to train your model.")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Natural language · LSTM inference</div>
      <h1>Next Word Predictor</h1>
      <p>Give your trained language model a phrase and see what it predicts next.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = [path.name for _, path in required_files if not path.is_file()]
if missing:
    st.warning(
        "Add the missing model files beside `app.py` to enable predictions: "
        + ", ".join(f"`{name}`" for name in missing)
        + "."
    )
    st.code(
        "lstm_next_word_streamlit/\n"
        "├── app.py\n"
        "├── tokenizer.pkl\n"
        "├── lstm_model.h5\n"
        "└── max_len.pkl",
        language="text",
    )
    st.stop()

try:
    tokenizer, model, saved_max_len, sequence_length, index_word = load_artifacts(
        str(TOKENIZER_PATH),
        str(MODEL_PATH),
        str(MAX_LEN_PATH),
        TOKENIZER_PATH.stat().st_mtime,
        MODEL_PATH.stat().st_mtime,
        MAX_LEN_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error("The model files could not be loaded.")
    st.exception(exc)
    st.info(
        "Check that the pickle files match this model, and that `lstm_model.h5` was saved in a compatible Keras/TensorFlow version."
    )
    st.stop()

vocab_size = len(getattr(tokenizer, "word_index", {}))
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Model", "LSTM")
metric_b.metric("Input window", f"{sequence_length} tokens")
metric_c.metric("Vocabulary", f"{vocab_size:,} words")

st.markdown("### Try a phrase")
with st.form("prediction_form"):
    seed_text = st.text_area(
        "Starting text",
        placeholder="Example: once upon a time",
        height=115,
        help="Enter the words you want the model to continue. The app keeps the most recent tokens if the phrase is longer than the model's input window.",
    )
    top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("✨ Predict next word", use_container_width=True)

if submitted:
    if not seed_text.strip():
        st.session_state.pop("prediction_result", None)
        st.warning("Enter a phrase first.")
    else:
        try:
            candidates, token_count = predict_next_words(
                seed_text.strip(), tokenizer, model, sequence_length, index_word, top_k
            )
            st.session_state["prediction_result"] = {
                "seed_text": seed_text.strip(),
                "candidates": candidates,
                "token_count": token_count,
            }
        except Exception as exc:
            st.session_state.pop("prediction_result", None)
            st.error(str(exc))

result = st.session_state.get("prediction_result")
if result:
    candidates = result["candidates"]
    best = candidates[0]
    st.markdown("---")
    st.markdown("### Prediction")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="word-card">
              <div class="word-label">Most likely next word</div>
              <div class="word-value">{best['Word']}</div>
              <div class="muted">Confidence: {best['Probability']:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**Your phrase + prediction**")
        st.info(f"{result['seed_text']} **{best['Word']}**")
        st.caption(
            f"Recognized {result['token_count']} token(s); the model uses the latest {sequence_length} token(s) as context."
        )
    with right:
        display = pd.DataFrame(candidates)
        display.insert(0, "Rank", range(1, len(display) + 1))
        display["Probability"] = display["Probability"].map(lambda value: f"{value:.2%}")
        st.markdown("**Other likely next words**")
        st.dataframe(display, hide_index=True, use_container_width=True)
        chart_data = pd.DataFrame(
            {"Word": [item["Word"] for item in candidates],
             "Probability": [item["Probability"] for item in candidates]}
        ).set_index("Word")
        st.bar_chart(chart_data, use_container_width=True)
else:
    st.markdown(
        """
        <div class="word-card">
          <div class="word-label">Ready when you are</div>
          <div class="word-value">Start typing…</div>
          <div class="muted">Your model's highest-ranked next word will appear here.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("Predictions depend on the training data and vocabulary of your saved LSTM model.")
