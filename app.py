import os
import time
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.feature_extraction import extract_features

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense — Speech Emotion AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 7 Emotions ────────────────────────────────────────────────────────────────
# Index order MUST match the label encoding used during training (see train_colab)
# 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
EMOTIONS = {
    0: ("Angry",    "😠", "#F85149", "There's intensity in your voice. Want to talk about what's bothering you?"),
    1: ("Disgust",  "🤢", "#7EE787", "Something seems to be really bothering you. Want to share what's going on?"),
    2: ("Fear",     "😨", "#BC8CFF", "I sense some anxiety or worry. Take a breath — you're safe here."),
    3: ("Happy",    "😊", "#F7C948", "Your voice carries such warm, joyful energy! What's made today good?"),
    4: ("Neutral",  "😐", "#8B949E", "You sound composed and steady. Anything on your mind?"),
    5: ("Sad",      "😢", "#58A6FF", "I can hear something heavy in your voice. That's okay — I'm here."),
    6: ("Surprise", "😲", "#FF9500", "Something unexpected happening? Tell me more!"),
}

CHATBOT_RESPONSES = {
    "Angry":    [
        "I can feel the intensity. It's okay to feel frustrated — what happened?",
        "Something's got you fired up. Want to vent? I'm not going anywhere.",
        "That energy is real. Let's talk through what's bothering you.",
    ],
    "Disgust":  [
        "Something clearly didn't sit right with you. Want to talk about it?",
        "That reaction is valid — what happened that felt so off-putting?",
        "I hear you. Sometimes things genuinely bother us for good reason. What's going on?",
    ],
    "Fear":     [
        "I sense some worry or anxiety. Take a deep breath — you're safe here. 🫂",
        "Fear can be overwhelming. What feels uncertain or scary right now?",
        "It sounds like something has you on edge. Want to talk it through together?",
    ],
    "Happy":    [
        "Your happiness is contagious! 🌟 What's made today good for you?",
        "Love that energy! Something exciting happening in your life?",
        "You sound great — want to share what's been going right?",
    ],
    "Neutral":  [
        "You sound steady. Anything on your mind you'd like to explore?",
        "A calm state is a great foundation — what would you like to work through today?",
        "You seem balanced. Want to talk or just have some background support?",
    ],
    "Sad":      [
        "I hear a heaviness in your voice. Take your time — I'm here to listen. 💙",
        "It sounds like something might be weighing on you. Want to talk about it?",
        "Sadness is valid. You don't have to navigate this alone. What's going on?",
    ],
    "Surprise": [
        "Whoa, something caught you off guard! Good surprise or not so much?",
        "You sound taken aback! What happened that surprised you?",
        "Life threw you a curveball? Tell me all about it!",
    ],
}

# ─── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0D1117;
        color: #E6EDF3;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1100px; }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #E6EDF3;
        letter-spacing: -0.5px;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #8B949E;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    .hero-accent {
        background: linear-gradient(90deg, #58A6FF, #BC8CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8B949E;
        margin-bottom: 0.75rem;
    }
    .emotion-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.75rem 1.25rem;
        border-radius: 50px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .bot-bubble {
        background: #1C2128;
        border: 1px solid #21262D;
        border-left: 3px solid #58A6FF;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        font-size: 1rem;
        line-height: 1.6;
        color: #CDD9E5;
        margin-top: 0.5rem;
    }
    .user-bubble {
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px 0 0 12px;
        border-right: 3px solid #BC8CFF;
        padding: 0.75rem 1.25rem;
        font-size: 0.95rem;
        color: #D1D5DB;
        margin-top: 0.5rem;
        text-align: right;
    }
    .divider { border: none; border-top: 1px solid #21262D; margin: 1.5rem 0; }
    .stButton > button {
        background: linear-gradient(135deg, #1F6FEB, #58A6FF);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600; font-size: 0.95rem;
        cursor: pointer; transition: opacity 0.2s; width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stFileUploader > div {
        border: 2px dashed #21262D !important;
        border-radius: 10px !important;
        background: #161B22 !important;
    }
    audio { width: 100%; border-radius: 8px; margin-top: 0.5rem; }
    .tip { font-size: 0.82rem; color: #6E7681; margin-top: 0.4rem; }
    .tag {
        display: inline-block; padding: 0.2rem 0.65rem;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
        margin-right: 0.3rem; background: #21262D; color: #8B949E;
    }

    /* Confidence tracker */
    @keyframes growBar { from { width: 0%; } }
    .conf-row {
        display: flex; align-items: center; gap: 0.75rem;
        margin-bottom: 0.55rem;
    }
    .conf-icon { font-size: 1.1rem; width: 24px; text-align: center; }
    .conf-name {
        width: 80px; font-size: 0.85rem; font-weight: 500; color: #CDD9E5;
        font-family: 'Space Grotesk', sans-serif;
    }
    .conf-track {
        flex: 1; height: 18px; background: #21262D;
        border-radius: 9px; overflow: hidden; position: relative;
    }
    .conf-fill {
        height: 100%; border-radius: 9px;
        animation: growBar 0.8s ease-out;
        display: flex; align-items: center; justify-content: flex-end;
        padding-right: 6px;
    }
    .conf-pct {
        width: 48px; text-align: right; font-size: 0.8rem;
        font-weight: 600; color: #8B949E; font-family: 'Space Grotesk', sans-serif;
    }
    .conf-row.top .conf-name, .conf-row.top .conf-pct { color: #E6EDF3; }
    .conf-row.top .conf-track { box-shadow: 0 0 0 1.5px rgba(255,255,255,0.15); }

    /* History pills */
    .history-strip {
        display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.5rem;
    }
    .history-pill {
        padding: 0.3rem 0.7rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
        border: 1px solid #21262D; background: #161B22;
        display: flex; align-items: center; gap: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_confidence_tracker(probs, predicted_idx):
    """Animated horizontal bars for all 7 emotions, sorted by confidence."""
    order = np.argsort(probs)[::-1]   # highest first
    rows_html = ""
    for rank, idx in enumerate(order):
        name, icon, color, _ = EMOTIONS[idx]
        pct = probs[idx] * 100
        is_top = "top" if idx == predicted_idx else ""
        # NOTE: built as a single-line string with no leading whitespace/newlines.
        # Multi-line triple-quoted f-strings with indentation get misread by
        # Streamlit's markdown parser as a code block, which is why the HTML
        # was leaking as raw visible text in the UI.
        rows_html += (
            f'<div class="conf-row {is_top}">'
            f'<span class="conf-icon">{icon}</span>'
            f'<span class="conf-name">{name}</span>'
            f'<div class="conf-track">'
            f'<div class="conf-fill" style="width:{pct:.1f}%; background:linear-gradient(90deg, {color}99, {color});"></div>'
            f'</div>'
            f'<span class="conf-pct">{pct:.1f}%</span>'
            f'</div>'
        )
    st.markdown(f'<div style="margin-top:0.5rem;">{rows_html}</div>', unsafe_allow_html=True)


def render_history_strip():
    """Show last 8 predictions as small pills."""
    if "prediction_log" not in st.session_state or not st.session_state.prediction_log:
        return
    pills = ""
    for entry in st.session_state.prediction_log[-8:][::-1]:
        name, icon, color, conf = entry
        pills += f'<div class="history-pill" style="border-color:{color}55;"><span>{icon}</span><span style="color:{color}">{name}</span><span style="color:#6E7681">{conf:.0f}%</span></div>'
    st.markdown(f'<div class="history-strip">{pills}</div>', unsafe_allow_html=True)


# ─── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "models/emotion_model.h5"
    if not os.path.exists(model_path):
        return None, f"Model not found at `{model_path}`."
    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


# ─── Waveform ──────────────────────────────────────────────────────────────────
def plot_waveform(audio_path, accent_color="#58A6FF"):
    y, sr = librosa.load(audio_path, duration=5)
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=(8, 1.8))
    fig.patch.set_facecolor("#161B22")
    ax.set_facecolor("#161B22")
    ax.fill_between(times, y, alpha=0.4, color=accent_color)
    ax.plot(times, y, color=accent_color, linewidth=0.6, alpha=0.9)
    ax.axhline(0, color="#21262D", linewidth=0.8)
    ax.set_xlabel("Time (s)", color="#8B949E", fontsize=8)
    ax.set_ylabel("Amplitude", color="#8B949E", fontsize=8)
    ax.tick_params(colors="#8B949E", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262D")
    plt.tight_layout(pad=0.4)
    return fig


# ─── Confidence chart ──────────────────────────────────────────────────────────
def plot_confidence(probs):
    labels = [EMOTIONS[i][0] for i in range(len(EMOTIONS))]
    colors = [EMOTIONS[i][2] for i in range(len(EMOTIONS))]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#161B22")
    ax.set_facecolor("#161B22")
    bars = ax.barh(labels, probs * 100, color=colors, alpha=0.85, height=0.55)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#8B949E", fontsize=8)
    ax.tick_params(colors="#E6EDF3", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262D")
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", color="#8B949E", fontsize=7.5)
    plt.tight_layout(pad=0.4)
    return fig


# ─── Prediction ────────────────────────────────────────────────────────────────
def predict(model, audio_path):
    features = extract_features(audio_path)
    if features is None:
        raise ValueError("Feature extraction returned None — audio may be corrupted or too short.")
    x = features.reshape(1, features.shape[0], 1)
    probs = model.predict(x, verbose=0)[0]
    return int(np.argmax(probs)), probs


# ─── Chat ──────────────────────────────────────────────────────────────────────
def init_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_emotion" not in st.session_state:
        st.session_state.last_emotion = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None   # (emotion_idx, probs, elapsed)
    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = []


def add_bot_message(emotion_name):
    import random
    responses = CHATBOT_RESPONSES.get(emotion_name, ["I'm here for you. Tell me more."])
    st.session_state.chat_history.append(("bot", random.choice(responses), emotion_name))


def render_chat():
    if not st.session_state.chat_history:
        st.markdown('<p class="tip">Upload audio and run analysis — the bot will respond based on your detected emotion.</p>', unsafe_allow_html=True)
        return
    last_emotion_shown = None
    for role, text, emotion in st.session_state.chat_history:
        em_entry = next((v for v in EMOTIONS.values() if v[0] == emotion), EMOTIONS[4])
        icon = em_entry[1]
        color = em_entry[2]

        # Show a small divider whenever a NEW audio analysis changes the detected emotion,
        # so repeated bot replies don't look like one continuous undifferentiated block.
        if role == "bot" and emotion != last_emotion_shown:
            st.markdown(
                f'<div style="font-size:0.7rem;color:{color};text-transform:uppercase;letter-spacing:1px;'
                f'margin:0.8rem 0 0.3rem 0;font-weight:600;">{icon} Detected: {emotion}</div>',
                unsafe_allow_html=True
            )
            last_emotion_shown = emotion

        if role == "bot":
            st.markdown(f'<div class="bot-bubble">{icon} {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="user-bubble">{text}</div>', unsafe_allow_html=True)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_chat()

    st.markdown("""
    <div class="hero-title">EmoSense <span class="hero-accent">Speech Emotion AI</span></div>
    <div class="hero-subtitle">Upload a voice recording — detects 7 emotions and responds with empathy.</div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading model…"):
        model, err = load_model()

    if err:
        st.error(f"⚠️ Could not load model: {err}")
        st.stop()

    # Show how many output neurons the loaded model has
    num_classes = model.output_shape[-1]
    st.markdown(
        f'<span class="tag">✓ Model ready</span> '
        f'<span class="tag">LSTM · {num_classes} emotions</span> '
        f'<span class="tag">MFCC + Chroma + Mel</span>',
        unsafe_allow_html=True
    )

    # Warn if old 6-class model is loaded
    if num_classes != 7:
        st.warning(
            f"⚠️ Loaded model has **{num_classes} output classes**, but this app expects **7**. "
            "Please retrain using the updated Colab notebook and replace `models/emotion_model.h5`."
        )

    st.markdown('<hr style="border:none;border-top:1px solid #21262D;margin:1.5rem 0">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card-title">🎵 Audio Input</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a WAV or MP3 file", type=["wav", "mp3"])

        if uploaded:
            suffix = ".wav" if uploaded.name.endswith(".wav") else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            st.audio(uploaded, format="audio/wav")

            st.markdown('<div class="card-title" style="margin-top:1rem;">📈 Waveform</div>', unsafe_allow_html=True)
            try:
                fig_wave = plot_waveform(tmp_path)
                st.pyplot(fig_wave, use_container_width=True)
                plt.close(fig_wave)
            except Exception as e:
                st.caption(f"Waveform unavailable: {e}")

            st.markdown('<hr style="border:none;border-top:1px solid #21262D;margin:1.5rem 0">', unsafe_allow_html=True)

            if st.button("🔍 Analyse Emotion"):
                if num_classes != 7:
                    st.error("Cannot predict — model has wrong number of output classes. Please retrain.")
                else:
                    with st.spinner("Extracting features and running inference…"):
                        t0 = time.time()
                        try:
                            emotion_idx, probs = predict(model, tmp_path)
                            elapsed = time.time() - t0

                            name, icon, color, desc = EMOTIONS[emotion_idx]
                            st.session_state.last_emotion = name
                            st.session_state.last_result = (emotion_idx, probs, elapsed)
                            st.session_state.prediction_log.append((name, icon, color, probs[emotion_idx] * 100))

                            add_bot_message(name)
                            st.rerun()

                        except Exception as e:
                            st.error(f"⚠️ Prediction failed: {e}")

            # Persistent result block — survives reruns (e.g. when sending a chat message)
            if st.session_state.last_result:
                emotion_idx, probs, elapsed = st.session_state.last_result
                name, icon, color, desc = EMOTIONS[emotion_idx]

                st.markdown('<hr style="border:none;border-top:1px solid #21262D;margin:1.5rem 0">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="emotion-badge" style="background:{color}22;border:1.5px solid {color};">'
                    f'<span style="font-size:1.8rem">{icon}</span>'
                    f'<span style="color:{color}">{name}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<p style="color:#8B949E;font-size:0.9rem">{desc}</p>', unsafe_allow_html=True)
                st.caption(f"Inference time: {elapsed:.2f}s · Confidence: {probs[emotion_idx]*100:.1f}%")

                st.markdown('<div class="card-title" style="margin-top:1rem;">📊 Confidence Tracker</div>', unsafe_allow_html=True)
                render_confidence_tracker(probs, emotion_idx)

                st.markdown('<div class="card-title" style="margin-top:1.2rem;">🕓 Recent Predictions</div>', unsafe_allow_html=True)
                render_history_strip()

    with col_right:
        st.markdown('<div class="card-title">💬 Emotion-Aware Support Bot</div>', unsafe_allow_html=True)
        render_chat()

        if st.session_state.last_emotion:
            st.markdown('<hr style="border:none;border-top:1px solid #21262D;margin:1.5rem 0">', unsafe_allow_html=True)
            user_input = st.text_input("Type a message…", placeholder="How are you feeling?",
                                       key="user_msg", label_visibility="collapsed")
            if st.button("Send", key="send_btn"):
                if user_input.strip():
                    st.session_state.chat_history.append(("user", user_input.strip(), st.session_state.last_emotion))
                    add_bot_message(st.session_state.last_emotion)
                    st.rerun()
            if st.button("🗑 Clear chat", key="clear_btn"):
                st.session_state.chat_history = []
                st.session_state.last_emotion = None
                st.rerun()
        else:
            st.markdown('<p class="tip" style="margin-top:1.5rem;">Analyse audio first to activate the support bot.</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()