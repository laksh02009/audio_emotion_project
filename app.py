import streamlit as st
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode

SAMPLE_RATE = 22050

# Load the trained model
with open("audio_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Audio Emotion Detection", layout="centered")

# CSS for dark theme and styling
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

# Initialize audio buffer and prediction in session state
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Setup WebRTC streamer for mic audio capture
webrtc_ctx = webrtc_streamer(
    key="audio-capture",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Function to fetch audio frames from the audio_receiver queue
def get_audio_frames(ctx):
    frames = []
    if ctx and ctx.audio_receiver:
        while True:
            try:
                frame = ctx.audio_receiver.get_frame(timeout=0.01)
                if frame is None:
                    break
                frames.append(frame)
            except Exception:
                break
    return frames

# Buffer audio frames continuously when mic is active
if webrtc_ctx.state.playing:
    frames = get_audio_frames(webrtc_ctx)
    for frame in frames:
        audio_np = frame.to_ndarray(format="flt32")  # shape: (channels, samples)
        st.session_state.audio_buffer.append(audio_np)

if webrtc_ctx.state.playing:
    st.success("✅ Microphone connected and streaming!")
else:
    st.info("🎙 Please allow microphone access in your browser.")

if not webrtc_ctx.state.playing:
    st.markdown(
        '<div class="warning-box">Microphone stream not connected yet. '
        'Please allow microphone access in your browser (click the lock icon → Allow). '
        'If it still fails, your network may block WebRTC or you may need a TURN server.</div>',
        unsafe_allow_html=True,
    )

# When user clicks Analyze button: process buffered audio & predict
if st.button("Analyze Recording"):
    if not st.session_state.audio_buffer:
        st.warning("No audio recorded yet! Please speak into the mic before analyzing.")
    else:
        # Concatenate buffered frames along time axis (channels, samples)
        audio_all = np.concatenate(st.session_state.audio_buffer, axis=1)

        # Convert multi-channel to mono by averaging channels
        if audio_all.shape[0] > 1:
            audio_all = np.mean(audio_all, axis=0)
        else:
            audio_all = audio_all.flatten()

        # Resample from 48kHz (typical WebRTC) to your SAMPLE_RATE 22050
        try:
            audio_resampled = librosa.resample(audio_all, orig_sr=48000, target_sr=SAMPLE_RATE)
        except Exception:
            audio_resampled = audio_all

        # Save temp wav file for feature extraction and playback
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, audio_resampled, SAMPLE_RATE)
            wav_path = tmp_wav.name

        # Playback recorded audio
        st.audio(wav_path, format="audio/wav")

        # Extract MFCC features (40 coefficients)
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)

        # Predict emotion
        try:
            prediction = model.predict(mfccs)[0]
            st.session_state.prediction = prediction
            st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

        # Clear audio buffer for next recording
        st.session_state.audio_buffer = []

# Show last prediction if available
if st.session_state.prediction:
    st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{st.session_state.prediction}</strong></div>', unsafe_allow_html=True)

