import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import re
import html
import base64
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn

from typing import Set
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.utils import logging
from lime.lime_text import LimeTextExplainer

logging.set_verbosity_error()

st.set_page_config(
    page_title="Arabic Hate Speech Demo",
    page_icon="logo.png",
    layout="wide"
)

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

LOGO_PATH = "logo.png"
logo_base64 = image_to_base64(LOGO_PATH)

st.markdown("""
<style>
:root {
    --dark-navy: #0B1F33;
    --teal: #00B4A6;
    --soft-cyan: #4FD1C5;
    --white: #FFFFFF;
    --light-gray: #E5E7EB;
    --matte-beige: #EFE7DA;
    --soft-beige: #F7F1E8;
    --danger: #D95F5F;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(79,209,197,.22), transparent 32%),
        radial-gradient(circle at bottom right, rgba(0,180,166,.18), transparent 34%),
        linear-gradient(135deg, #EFE7DA 0%, #F7F1E8 45%, #E9DDCF 100%);
    color: var(--dark-navy);
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

.hero-card {
    padding: 2rem 2.4rem;
    border-radius: 30px;
    background: rgba(255, 255, 255, .72);
    border: 1px solid rgba(11, 31, 51, .12);
    box-shadow: 0 18px 45px rgba(11, 31, 51, .13), 0 0 38px rgba(0, 180, 166, .16);
    margin-bottom: 1.5rem;
    backdrop-filter: blur(14px);
}

.hero-title-row {
    display: flex;
    align-items: center;
    gap: 1.4rem;
}

.hero-logo {
    width: 150px;
    height: 150px;
    object-fit: contain;
    flex-shrink: 0;
}

.hero-title {
    font-size: 2.7rem;
    line-height: 1.12;
    font-weight: 900;
    background: linear-gradient(90deg, #0B1F33, #00B4A6, #4FD1C5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #334155;
    font-size: 1.1rem;
    margin-top: .2rem;
    margin-left: 174px;
}

.result-card {
    padding: 1.5rem;
    border-radius: 26px;
    background: rgba(255, 255, 255, .74);
    border: 1px solid rgba(11, 31, 51, .12);
    box-shadow: 0 16px 35px rgba(11, 31, 51, .10), 0 0 24px rgba(0, 180, 166, .12);
    min-height: 170px;
    transition: all .25s ease;
}

.result-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 22px 45px rgba(11, 31, 51, .14), 0 0 32px rgba(0, 180, 166, .20);
}

.result-label {
    color: #475569;
    font-weight: 800;
}

.result-value {
    font-size: 2.4rem;
    font-weight: 900;
    margin-top: .45rem;
}

.hate-glow {
    color: var(--danger);
    text-shadow: 0 0 14px rgba(217, 95, 95, .28);
}

.safe-glow {
    color: var(--teal);
    text-shadow: 0 0 14px rgba(0, 180, 166, .30);
}

.cyan-glow {
    color: var(--soft-cyan);
    text-shadow: 0 0 14px rgba(79, 209, 197, .32);
}

.purple-glow {
    color: var(--teal);
    text-shadow: 0 0 14px rgba(0, 180, 166, .28);
}

.confidence {
    margin-top: .7rem;
    color: #334155;
}

.section-title {
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-size: 1.7rem;
    font-weight: 900;
    color: var(--dark-navy);
}

.lime-card {
    padding: 1.2rem;
    border-radius: 22px;
    background: rgba(255, 255, 255, .74);
    border: 1px solid rgba(0, 180, 166, .22);
    box-shadow: 0 14px 32px rgba(11, 31, 51, .10), 0 0 18px rgba(79, 209, 197, .12);
    margin-bottom: 1rem;
}

.lime-title {
    font-size: 1.25rem;
    font-weight: 900;
    margin-bottom: .2rem;
    color: var(--dark-navy);
}

.lime-sub {
    color: #475569;
    margin-bottom: .7rem;
}

textarea {
    background-color: rgba(255, 255, 255, .82) !important;
    color: #0B1F33 !important;
    border-radius: 20px !important;
    border: 1px solid rgba(0, 180, 166, .40) !important;
}

.stTextArea label {
    color: #0B1F33 !important;
    font-weight: 800 !important;
}

.stButton > button {
    border-radius: 18px;
    border: 1px solid rgba(0, 180, 166, .55);
    background: linear-gradient(90deg, #0B1F33, #00B4A6);
    color: #FFFFFF;
    font-weight: 900;
    min-height: 3.2rem;
    box-shadow: 0 12px 28px rgba(11, 31, 51, .18), 0 0 22px rgba(0, 180, 166, .20);
}

.stButton > button:hover {
    border: 1px solid rgba(79, 209, 197, .85);
    box-shadow: 0 18px 38px rgba(11, 31, 51, .22), 0 0 30px rgba(0, 180, 166, .28);
    transform: translateY(-2px);
}

.stProgress > div > div > div > div {
    background-image: linear-gradient(90deg, #00B4A6, #4FD1C5);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1F33 0%, #102A43 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "camelbert_mix_multilabel_with_emoji_feature"
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128

tasks = {
    0: ["Egyptian", "Saudi"],
    1: ["Not Hate", "Hate"],
    2: ["Political", "Religious"]
}

task_names = {
    0: "Dialect",
    1: "Hate",
    2: "Topic"
}

HATE_EMOJIS = set([
    '💦', '🐖', '🐷', '🐽', '👞', '🐕', '🐶', '💩', '🐄', '🐮',
    '🐑', '🐏', '👎', '😡', '🤬', '👺', '👿', '😠'
])

def emoji_flag(text):
    return 1 if any(e in text for e in HATE_EMOJIS) else 0


HATE_EMOJI_WHITELIST: Set[str] = HATE_EMOJIS

DEFAULT_AR_STOPWORDS: Set[str] = {
    "في","على","الى","إلى","من","عن","مع","حتى","و","او","أو","ثم","لكن","بل",
    "هذا","هذه","ذلك","تلك","هنا","هناك","ما","ماذا","لماذا","كيف","هل",
    "كان","كانت","يكون","تكون","ان","إن","أن","انا","أنت","انتي","هو","هي","هم","هن",
    "لما","كل","أي","اي","اذا","إذا"
}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
UNDERSCORE_RE = re.compile(r"_+")
EN_RE = re.compile(r"[A-Za-z]+")
DIGITS_RE = re.compile(r"[0-9\u0660-\u0669]+")
AR_LETTERS_RE = re.compile(
    r"[^ \n\t0-9\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)
EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF\U00002700-\U000027BF\U00002600-\U000026FF]"
)
TATWEEL_RE = re.compile(r"ـ")
DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
REPEAT_RE = re.compile(r"(.)\1{2,}")

def normalize_arabic(text: str) -> str:
    text = TATWEEL_RE.sub("", text)
    text = DIACRITICS_RE.sub("", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = REPEAT_RE.sub(r"\1", text)
    return text

def keep_hate_emojis_only(text: str) -> str:
    def _f(m):
        ch = m.group(0)
        return ch if ch in HATE_EMOJI_WHITELIST else ""
    return EMOJI_RE.sub(_f, text)

def remove_noise(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", "")
    text = UNDERSCORE_RE.sub(" ", text)
    text = EN_RE.sub(" ", text)
    text = DIGITS_RE.sub(" ", text)

    protected = {}
    for i, emoji in enumerate(HATE_EMOJIS):
        placeholder = f" رمزكراهيه{i} "
        if emoji in text:
            text = text.replace(emoji, placeholder)
            protected[f"رمزكراهيه{i}"] = emoji

    text = EMOJI_RE.sub(" ", text)
    text = AR_LETTERS_RE.sub(" ", text)
    text = re.sub(r"[؟?،؛:!.\-ـ…\"'“”‘’()\[\]{}]", " ", text)
    for placeholder, emoji in protected.items():
        text = text.replace(placeholder, emoji)

    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(text: str) -> str:
    return " ".join(t for t in text.split() if t not in DEFAULT_AR_STOPWORDS)

def clean_text(x) -> str:
    text = "" if x is None else str(x)
    text = normalize_arabic(text)
    text = remove_noise(text)
    text = remove_stopwords(text)
    return text


class MultiLabelCAMeLBERT(nn.Module):
    def __init__(self, model_path, dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_config(config)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + 1, 3)
        self.emoji_boost = nn.Parameter(torch.tensor(0.75, dtype=torch.float32))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, emoji_flag=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )

        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
        pooled = self.drop(pooled)

        if emoji_flag is None:
            emoji_flag = torch.zeros(pooled.size(0), device=pooled.device, dtype=pooled.dtype)
        else:
            emoji_flag = emoji_flag.to(pooled.dtype)

        x = torch.cat([pooled, emoji_flag.unsqueeze(1)], dim=1)
        logits = self.classifier(x)
        logits[:, 1] = logits[:, 1] + self.emoji_boost * emoji_flag

        return {"logits": logits}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = MultiLabelCAMeLBERT(MODEL_PATH)
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_PATH, "pytorch_model.bin"),
            map_location=device
        )
    )
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_raw(text):
    model.eval()
    cleaned_text = clean_text(text)

    enc = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    eflag = torch.tensor([float(emoji_flag(text))],device=device)

    with torch.no_grad():
        out = model(**enc, emoji_flag=eflag)

    probs = torch.sigmoid(out["logits"]).squeeze(0).cpu().numpy()

    return {
        "Dialect": {
            "Prediction": "Saudi" if probs[0] >= 0.5 else "Egyptian",
            "Egyptian": float(1 - probs[0]),
            "Saudi": float(probs[0])
        },
        "Hate": {
            "Prediction": "Hate" if probs[1] >= 0.5 else "Not Hate",
            "Not Hate": float(1 - probs[1]),
            "Hate": float(probs[1])
        },
        "Topic": {
            "Prediction": "Religious" if probs[2] >= 0.5 else "Political",
            "Political": float(1 - probs[2]),
            "Religious": float(probs[2])
        },
        "emoji_flag": int(emoji_flag(cleaned_text))
    }

def predict_proba_label(label_index):
    def classifier_fn(texts):
        texts = [clean_text(t) for t in list(texts)]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        eflags = torch.tensor(
            [float(emoji_flag(text)) for text in texts],
            device=device
        )

        with torch.no_grad():
            out = model(**enc, emoji_flag=eflags)

        probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
        p = probs[:, label_index]

        return np.column_stack([1 - p, p])

    return classifier_fn

def explain_text_live_streamlit(text):
    cleaned_text = clean_text(text)
    pred = predict_raw(text)

    if len(cleaned_text.split()) < 2:
        return pred, None

    selected_classes = {
        0: 1 if pred["Dialect"]["Prediction"] == "Saudi" else 0,
        1: 1 if pred["Hate"]["Prediction"] == "Hate" else 0,
        2: 1 if pred["Topic"]["Prediction"] == "Religious" else 0
    }

    explanations = {}

    for label_index, class_pair in tasks.items():
        class_id = selected_classes[label_index]
        class_name = class_pair[class_id]

        explainer = LimeTextExplainer(
            class_names=class_pair,
            split_expression=r"\s+",
            random_state=42
        )

        exp = explainer.explain_instance(
            text_instance=cleaned_text,
            classifier_fn=predict_proba_label(label_index),
            labels=[class_id],
            num_features=3,
            num_samples=60
        )

        df = pd.DataFrame(exp.as_list(label=class_id), columns=["Feature", "Weight"])
        df["Abs"] = df["Weight"].abs()
        df = df.sort_values("Abs", ascending=False).drop(columns="Abs").head(3).reset_index(drop=True)
        df["Direction"] = df["Weight"].apply(lambda w: "supports" if w > 0 else "opposes")

        explanations[label_index] = {
            "task": task_names[label_index],
            "class_name": class_name,
            "data": df
        }

    return pred, explanations

def lime_table_component(df):
    rows = ""

    for _, row in df.iterrows():
        direction = str(row["Direction"])
        direction_class = "supports" if direction == "supports" else "opposes"
        feature = html.escape(str(row["Feature"]))
        weight = round(float(row["Weight"]), 4)

        rows += f"""
        <tr>
            <td class="feature">{feature}</td>
            <td>{weight}</td>
            <td class="{direction_class}">{direction}</td>
        </tr>
        """

    table_html = f"""
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                background: transparent;
                font-family: Inter, Arial, sans-serif;
            }}
            table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                overflow: hidden;
                border-radius: 18px;
                background: #FFFFFF;
                border: 1px solid rgba(11, 31, 51, 0.12);
                box-shadow: 0 14px 28px rgba(11, 31, 51, 0.12);
            }}
            th {{
                background: #0B1F33;
                color: #FFFFFF;
                padding: 14px;
                text-align: left;
                font-size: 15px;
            }}
            td {{
                color: #0B1F33;
                padding: 14px;
                font-size: 15px;
                border-bottom: 1px solid rgba(11, 31, 51, 0.10);
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            .feature {{
                color: #0B1F33;
                font-weight: 800;
                text-align: right;
                direction: rtl;
            }}
            .supports {{
                color: #00B4A6;
                font-weight: 900;
            }}
            .opposes {{
                color: #D95F5F;
                font-weight: 900;
            }}
        </style>
    </head>
    <body>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Weight</th>
                    <th>Direction</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """

    components.html(table_html, height=190, scrolling=False)

def set_next_example(example_type):

    Political_examples = [
        "يا بلحه اتق الله بقى مش طبيعي اللي بتعملو في شعبك ربنا حسيبك",
        "البلد دي حالها صعبان عليه بس انا لازم اسيبها وامشي عشان يبقى ليا مستقبل",
        "نسبة العطالة عندنا لازم يتم النظر فيها شبابنا فيهم الخير و يستحقون افضل المناصب",
        "اقول حدك عاد ما ساعد بلدكم غيرنا يا خونه دولتنا خط احمر"
    ]

    Religious_examples = [
        "تدعي إنك وطني وأنت تحارب شرع الله وتستهزئ بالثوابت! ترا اللي يبيع دينه عشان يرضي الناس هذا منافق ومع الوقت بتنفضحون يا أذناب العلمانية",
        "حنا نحترم الناس كلهم وديننا علّمنا الأخلاق وحسن التعامل مهما اختلفت الآراء",
        "الحمد لله الدين يدعو للتسامح والرحمة بين الناس",
        "الجماعة دي متشددة وما عندهم أي احترام للناس"
    ]

    Different_Levels_examples = [
    "وربنا اللي بيحصل ده مش طبيعي خالص واللي بيحصل في البلد محتاج وقفة جد 😡",

    "يا أذناب المحور الثلاثي تظنون إنكم بتهزون جبل حنا أدرى بخبثكم يا 🐕 تلبسون ثوب الدين وأنتم 👿 في هيئة بشر السعودية خط أحمر والزمن بيكشف خيانتكم يا عبيد الصهاينة",

    "يا ليت المسؤولين يسمعون صوت الشعب ويشوفون الناس كيف عايشة الله يكون بالعون 😠"
]

    if example_type == "political":
        examples = Political_examples

    elif example_type == "religious":
        examples = Religious_examples

    elif example_type == "Different_Levels":
        examples = Different_Levels_examples

    else:
        return

    key = f"{example_type}_index"

    if key not in st.session_state:
        st.session_state[key] = 0

    st.session_state["tweet_text"] = examples[st.session_state[key]]

    st.session_state[key] = (
        st.session_state[key] + 1
    ) % len(examples)

st.markdown(f"""
<div class="hero-card">
    <div class="hero-title-row">
        <img src="data:image/png;base64,{logo_base64}" class="hero-logo">
        <div>
            <div class="hero-title">Arabic Hate Speech Detection</div>
            <div class="hero-subtitle">AI-powered Arabic tweet analysis</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if "tweet_text" not in st.session_state:
    st.session_state["tweet_text"] = ""

with st.sidebar:

    st.markdown("---")
    st.markdown("### Ready Examples")

    if st.button(
        "Political Example",
        use_container_width=True,
        key="political_btn"
    ):
        set_next_example("Political")

    if st.button(
        "Religious Example",
        use_container_width=True,
        key="religious_btn"
    ):
        set_next_example("Religious")

    if st.button(
        "Different Levels Example",
        use_container_width=True,
        key="test_btn"
    ):
        set_next_example("Different_Levels")

st.text_area(
    "Enter Arabic Tweet",
    key="tweet_text",
    height=160
)

if st.button("Analyze", use_container_width=True):

    user_text = str(st.session_state["tweet_text"]).strip()
    cleaned_user_text = clean_text(user_text)

    if cleaned_user_text == "" and emoji_flag(user_text) == 0:
        st.warning("Text became empty after cleaning. Please enter another tweet.")

    else:
        with st.spinner("Analyzing..."):
            result, explanations = explain_text_live_streamlit(user_text)

        st.success("Analysis Complete")

        hate_label = result["Hate"]["Prediction"]
        dialect_label = result["Dialect"]["Prediction"]
        topic_label = result["Topic"]["Prediction"]

        hate_conf = result["Hate"][hate_label]
        dialect_conf = result["Dialect"][dialect_label]
        topic_conf = result["Topic"][topic_label]

        hate_class = "hate-glow" if hate_label == "Hate" else "safe-glow"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Hate Speech</div>
                <div class="result-value {hate_class}">{hate_label}</div>
                <div class="confidence">Confidence: {hate_conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Dialect</div>
                <div class="result-value cyan-glow">{dialect_label}</div>
                <div class="confidence">Confidence: {dialect_conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Topic</div>
                <div class="result-value purple-glow">{topic_label}</div>
                <div class="confidence">Confidence: {topic_conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Detailed Probabilities</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Hate Speech")
        st.progress(result["Hate"]["Hate"])
        st.write(f"Not Hate: {result['Hate']['Not Hate']:.3f}")
        st.write(f"Hate: {result['Hate']['Hate']:.3f}")

        st.markdown("### Dialect")
        st.progress(result["Dialect"]["Saudi"])
        st.write(f"Egyptian: {result['Dialect']['Egyptian']:.3f}")
        st.write(f"Saudi: {result['Dialect']['Saudi']:.3f}")

        st.markdown("### Topic")
        st.progress(result["Topic"]["Religious"])
        st.write(f"Political: {result['Topic']['Political']:.3f}")
        st.write(f"Religious: {result['Topic']['Religious']:.3f}")

        st.markdown("---")

        if result["emoji_flag"] == 1:
            st.warning("Hate-related emoji detected.")
        else:
            st.info("No hate-related emoji detected.")

        if explanations is not None:
            st.markdown(
                '<div class="section-title">Tweet Explanation</div>',
                unsafe_allow_html=True
            )

            lime_cols = st.columns(3)

            for idx, label_index in enumerate([0, 1, 2]):
                item = explanations[label_index]
                df = item["data"].copy()

                with lime_cols[idx]:
                    st.markdown(f"""
                    <div class="lime-card">
                        <div class="lime-title">{item["task"]} → {item["class_name"]}</div>
                        <div class="lime-sub">Top words influencing this prediction</div>
                    </div>
                    """, unsafe_allow_html=True)

                    lime_table_component(df)

        else:
            st.info("LIME explanation requires at least two words after cleaning. Prediction was generated without LIME explanation.")