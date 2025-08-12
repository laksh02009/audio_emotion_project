import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

SAMPLE_RATE = 22050
DURATION = 5  # seconds

# Load the trained model
with open("audio_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Audio Emotion Detection", layout="centered")

# ---- Custom CSS (your existing style) ----
st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .block-container {
        max-width: 650px;
        padding: 2rem 3rem;
        margin: auto;
    }
    h1 {
        font-weight: 700;
        color: #4dd0e1;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        color: #b0b0b0;
    }
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
    .countdown {
        text-align: center;
        font-weight: 600;
        font-size: 1.7rem;
        color: #4dd0e1;
        margin-bottom: 1rem;
    }
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
    div[data-testid="stAudio"] {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- App Header ----
st.markdown("<h1>🎙 Audio Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Record your voice and let the AI predict your emotion.</div>', unsafe_allow_html=True)

# ---- Store recorded audio frames ----
recorded_frames = []

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
    def recv_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        recorded_frames.append(audio)
        return frame

# ---- Recording Logic ----
if st.button("Start Recording"):
    st.markdown('<div class="countdown">Recording will start in 3 seconds...</div>', unsafe_allow_html=True)
    time.sleep(1)
    for i in range(3, 0, -1):
        st.markdown(f'<div class="countdown">{i}</div>', unsafe_allow_html=True)
        time.sleep(1)

    st.markdown('<div class="countdown">🎤 Recording...</div>', unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="audio-capture",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioProcessor
    )

# ---- Process Recorded Audio ----
if st.button("Analyze Recording"):
    if not recorded_frames:
        st.warning("No audio recorded yet!")
    else:
        # Combine frames and save as WAV
        audio_np = np.concatenate(recorded_frames, axis=0).astype(np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio_np, SAMPLE_RATE)
            temp_path = tmp.name

        # Play audio
        st.audio(temp_path, format="audio/wav")

        # Feature extraction
        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)

        # Prediction
        prediction = model.predict(mfccs)[0]
        st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{prediction}</strong></div>', unsafe_allow_html=True)

        # Clear buffer
        recorded_frames.clear()
