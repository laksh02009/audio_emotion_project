import streamlit as st
import sounddevice as sd
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
import time

SAMPLE_RATE = 22050
DURATION = 5  # seconds

# Load the trained model
with open("audio_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Audio Emotion Detection", layout="centered")

# Custom CSS for a subtle, clean dark theme
st.markdown(
    """
    <style>
    /* Background and text */
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Container padding and width */
    .block-container {
        max-width: 650px;
        padding: 2rem 3rem;
        margin: auto;
    }
    /* Title styling */
    h1 {
        font-weight: 700;
        color: #4dd0e1;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        color: #b0b0b0;
    }
    /* Button style */
    div.stButton > button {
        background-color: #4dd0e1;
        color: #1e1e1e;
        font-weight: 600;
        padding: 0.5rem 1.6rem;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
        width: 100%;
        font-size: 1.05rem;
        cursor: pointer;
        box-shadow: 0 3px 6px rgba(77, 208, 225, 0.4);
    }
    div.stButton > button:hover {
        background-color: #26c6da;
        box-shadow: 0 5px 12px rgba(38, 198, 218, 0.6);
    }
    /* Countdown text */
    .countdown {
        text-align: center;
        font-weight: 600;
        font-size: 1.7rem;
        color: #4dd0e1;
        margin-bottom: 1rem;
    }
    /* Prediction box */
    .prediction-box {
        background-color: #2c2c2c;
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin-top: 2rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #4dd0e1;
        box-shadow: 0 0 12px rgba(77, 208, 225, 0.5);
    }
    /* Audio player centered */
    div[data-testid="stAudio"] {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & subtitle
st.markdown("<h1>🎙 Audio Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Record your voice and let the AI predict your emotion.</div>', unsafe_allow_html=True)

if st.button("Start Recording"):
    st.markdown('<div class="countdown">Recording will start in 3 seconds...</div>', unsafe_allow_html=True)
    time.sleep(1)
    for i in range(3, 0, -1):
        st.markdown(f'<div class="countdown">{i}</div>', unsafe_allow_html=True)
        time.sleep(1)

    st.markdown('<div class="countdown">🎤 Recording...</div>', unsafe_allow_html=True)
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    st.markdown('<div class="countdown">✅ Recording complete!</div>', unsafe_allow_html=True)

    # Save temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, SAMPLE_RATE)

    # Play back recorded audio (centered by CSS)
    st.audio(temp_file.name, format="audio/wav")

    # Feature extraction & prediction
    y, sr = librosa.load(temp_file.name, sr=SAMPLE_RATE)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)
    prediction = model.predict(mfccs)[0]

    st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
