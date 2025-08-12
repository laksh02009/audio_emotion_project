import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("Mic Test")

webrtc_ctx = webrtc_streamer(
    key="mic-test",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.state.playing:
    st.success("Mic streaming!")
else:
    st.info("Please allow mic access")


# Show prediction if available
if st.session_state.prediction:
    st.markdown(f'<div class="prediction-box">🎯 Predicted Emotion: <strong>{st.session_state.prediction}</strong></div>', unsafe_allow_html=True)
