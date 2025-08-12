# app.py
import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import av

SAMPLE_RATE = 22050

# Load the trained model
with open("audio_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Audio Emotion Detection", layout="centered")

# ---- CSS unchanged ----
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

st.markdown("<h1>🎙 Audio Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Record your voice and let the AI predict your emotion.</div>', unsafe_allow_html=True)

# ---- session buffer ----
if "recorded_frames" not in st.session_state:
    st.session_state.recorded_frames = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ---- TURN/STUN config (keep your Xirsys values) ----
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:bn-turn2.xirsys.com"]},
        {
            "urls": [
                "turn:bn-turn2.xirsys.com:80?transport=udp",
                "turn:bn-turn2.xirsys.com:3478?transport=udp",
                "turn:bn-turn2.xirsys.com:80?transport=tcp",
                "turn:bn-turn2.xirsys.com:3478?transport=tcp",
                "turns:bn-turn2.xirsys.com:443?transport=tcp",
                "turns:bn-turn2.xirsys.com:5349?transport=tcp"
            ],
            "username": "FsPFEnE5TE2ckMScYc3pVC22O8kJ2AfIR8qGUSlqL7-SN2E-GuGbi4p_zLf3CZlPAAAAAGibaRxsYWtzaDAyMDA5",
            "credential": "d998be8c-7797-11f0-9520-0242ac140004"
        }
    ]
}

# ---- start webRTC (no audio_processor_factory: we'll pull frames from audio_receiver) ----
webrtc_ctx = webrtc_streamer(
    key="audio-capture",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# ---- read queued frames safely using available API ----
def drain_audio_receiver(ctx):
    """
    Pull available frames from ctx.audio_receiver using supported methods.
    Returns list of av.AudioFrame objects.
    """
    frames = []
    if not ctx:
        return frames
    recv = getattr(ctx, "audio_receiver", None)
    if not recv:
        return frames

    # Try get_frame (blocking with timeout)
    get_frame = getattr(recv, "get_frame", None)
    if callable(get_frame):
        # collect frames until timeout
        while True:
            try:
                frame = recv.get_frame(timeout=0.01)  # small timeout
                if frame is None:
                    break
                frames.append(frame)
            except Exception:
                # timeout or no more frames
                break
        return frames

    # Fallback: try recv() (may be blocking) — do one non-blocking attempt
    recv_fn = getattr(recv, "recv", None)
    if callable(recv_fn):
        try:
            frame = recv_fn(timeout=0.01)  # some implementations accept timeout
            if frame is not None:
                frames.append(frame)
        except TypeError:
            # recv may not accept timeout — call once and catch exceptions
            try:
                frame = recv_fn()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass
        except Exception:
            pass

    return frames

# ---- collect frames into session buffer ----
if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
    # drain any frames available this run
    frames = drain_audio_receiver(webrtc_ctx)
    for f in frames:
        try:
            audio = f.to_ndarray()
            st.session_state.recorded_frames.append(audio)
        except Exception:
            # skip bad frames
            continue

# ---- UI status ----
if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
    st.success("✅ Microphone is connected and streaming!")
else:
    st.info("🎙 Please allow microphone access in your browser or wait for connection.")

if not (webrtc_ctx and getattr(webrtc_ctx.state, "playing", False)):
    st.markdown(
        '<div class="warning-box">Microphone stream not connected yet. '
        'Please allow microphone access in your browser (click the lock icon → Allow). '
        'If it still fails, your network may block WebRTC and you may need a TURN server.</div>',
        unsafe_allow_html=True,
    )
else:
    st.success("Microphone connected — speak now (your audio is being captured).")

# ---- Live prediction display (optional) ----
if st.session_state.prediction is not None:
    st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{st.session_state.prediction}</strong></div>', unsafe_allow_html=True)

# ---- Analyze Button: merge buffered frames, resample, extract features, predict ----
if st.button("Analyze Recording"):
    if not st.session_state.recorded_frames:
        st.warning("No audio recorded yet! Please speak into the mic before analyzing.")
    else:
        # concatenate frames (frames from webRTC are typically shape (channels, samples))
        audio_np = np.concatenate(st.session_state.recorded_frames, axis=0).astype(np.float32)

        # Many browsers send 48k sampling. Resample to SAMPLE_RATE
        # Normalize if necessary: frames may already be float in -1..1 or int16 -32768..32767
        # We'll try to detect scale
        if audio_np.dtype == np.int16 or audio_np.max() > 1.0:
            audio_np = audio_np / 32768.0

        # flatten to mono if multi-channel
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0)

        # resample from 48000 (typical WebRTC) to your SAMPLE_RATE
        try:
            audio_resampled = librosa.resample(audio_np, orig_sr=48000, target_sr=SAMPLE_RATE)
        except Exception:
            # fallback: assume it's already at SAMPLE_RATE
            audio_resampled = audio_np

        # Save to wav (for playback and debugging)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio_resampled, SAMPLE_RATE)
            tmp_path = tmp.name

        st.audio(tmp_path, format="audio/wav")

        # feature extraction (match training: n_mfcc=40)
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)

        # predict
        try:
            prediction = model.predict(mfccs)[0]
            st.session_state.prediction = prediction
            st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

        # clear buffer for next recording
        st.session_state.recorded_frames.clear()
