import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

SAMPLE_RATE = 22050

# Load the trained model
with open("audio_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Audio Emotion Detection", layout="centered")

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    .main { background-color: #1e1e1e; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .block-container { max-width: 650px; padding: 2rem 3rem; margin: auto; }
    h1 { font-weight: 700; color: #4dd0e1; text-align: center; margin-bottom: 0.2rem; }
    .subtitle { text-align: center; font-size: 1.1rem; margin-bottom: 2rem; color: #b0b0b0; }
    div.stButton > button { background-color: #4dd0e1; color: #1e1e1e; font-weight: 600; padding: 0.5rem 1.6rem; border-radius: 8px; border: none; transition: background-color 0.3s ease; width: 100%; font-size: 1.05rem; cursor: pointer; box-shadow: 0 3px 6px rgba(77, 208, 225, 0.4); }
    div.stButton > button:hover { background-color: #26c6da; box-shadow: 0 5px 12px rgba(38, 198, 218, 0.6); }
    .prediction-box { background-color: #2c2c2c; border-radius: 12px; padding: 1.6rem 2rem; margin-top: 2rem; text-align: center; font-size: 1.4rem; font-weight: 700; color: #4dd0e1; box-shadow: 0 0 12px rgba(77, 208, 225, 0.5); }
    div[data-testid="stAudio"] { display: flex; justify-content: center; margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- App Header ----
st.markdown("<h1>🎙 Audio Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Record your voice and let the AI predict your emotion.</div>', unsafe_allow_html=True)

# ---- Store recorded audio frames ----
if "recorded_frames" not in st.session_state:
    st.session_state.recorded_frames = []

class AudioProcessor(AudioProcessorBase):
    def recv_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        st.session_state.recorded_frames.append(audio)
        return frame

# ---- Recording UI ----
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},  # Public Google STUN
        # TURN server example (paid or free trial):
        # {
        #     "urls": ["turn:YOUR_TURN_SERVER_IP:3478"],
        #     "username": "YOUR_USERNAME",
        #     "credential": "YOUR_PASSWORD"
        # }
    ]
}

webrtc_ctx = webrtc_streamer(
    key="audio-capture",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
    rtc_configuration=RTC_CONFIGURATION
)

# ---- Analyze Button ----
if st.button("Analyze Recording"):
    if not st.session_state.recorded_frames:
        st.warning("No audio recorded yet! Please speak into the mic before analyzing.")
    else:
        audio_np = np.concatenate(st.session_state.recorded_frames, axis=0).astype(np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio_np, SAMPLE_RATE)
            temp_path = tmp.name

        st.audio(temp_path, format="audio/wav")

        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)
        prediction = model.predict(mfccs)[0]

        st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{prediction}</strong></div>', unsafe_allow_html=True)

        st.session_state.recorded_frames.clear()

