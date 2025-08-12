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

# ---- Custom CSS (keep yours as before) ----
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
    .warning-box { background: #3a3a3a; padding: 0.8rem; border-radius: 8px; color: #ffd54f; margin-bottom: 1rem; text-align:center;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- App Header ----
st.markdown("<h1>🎙 Audio Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Record your voice and let the AI predict your emotion.</div>', unsafe_allow_html=True)

# ---- Session storage for frames ----
if "recorded_frames" not in st.session_state:
    st.session_state.recorded_frames = []

# AudioProcessor with safe guard to prevent server-side crash
class AudioProcessor(AudioProcessorBase):
    def recv_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            audio = frame.to_ndarray()
        except Exception as e:
            # If conversion fails, skip this frame (prevents aioice/socket crashes bubbling up)
            # you can log e to server logs if you want
            return frame
        # append frames into session state list
        st.session_state.recorded_frames.append(audio)
        return frame

# ---- RTC config with STUN (you already had this; keep TURN if you have one) ----
rtc_configuration = {
    "iceServers": [
        {
            "urls": ["stun:bn-turn2.xirsys.com"]
        },
        {
            "username": "FsPFEnE5TE2ckMScYc3pVC22O8kJ2AfIR8qGUSlqL7-SN2E-GuGbi4p_zLf3CZlPAAAAAGibaRxsYWtzaDAyMDA5",
            "credential": "d998be8c-7797-11f0-9520-0242ac140004",
            "urls": [
                "turn:bn-turn2.xirsys.com:80?transport=udp",
                "turn:bn-turn2.xirsys.com:3478?transport=udp",
                "turn:bn-turn2.xirsys.com:80?transport=tcp",
                "turn:bn-turn2.xirsys.com:3478?transport=tcp",
                "turns:bn-turn2.xirsys.com:443?transport=tcp",
                "turns:bn-turn2.xirsys.com:5349?transport=tcp"
            ]
        }
    ]
}

st.title("🎤 Microphone Test with TURN Server")

webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,  # Only sending audio
    audio_receiver_size=256,
    rtc_configuration=rtc_configuration,  # <-- Xirsys config here
    media_stream_constraints={
        "audio": True,
        "video": False
    }
)

# ---- Connection status checks & user hints ----
connected = False
# webrtc_ctx may be None or have state attribute; handle both
if webrtc_ctx is not None:
    try:
        # .state.playing is a reliable check to see if WebRTC is up
        connected = bool(getattr(webrtc_ctx.state, "playing", False))
    except Exception:
        connected = False

if not connected:
    st.markdown(
        '<div class="warning-box">Microphone stream not connected yet. '
        'Please allow microphone access in your browser (click the lock icon → Allow). '
        'If it still fails, your network may block WebRTC and you may need a TURN server.</div>',
        unsafe_allow_html=True,
    )
else:
    st.success("Microphone connected — speak now (your audio is being captured).")

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
