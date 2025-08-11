import sounddevice as sd
import numpy as np
import pickle
import librosa
import time
import os
import winsound
import soundfile as sf  # <-- for saving audio

DURATION = 10  # seconds
SAMPLE_RATE = 22050  # Hz

# Load the trained model
with open("audio_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Countdown
print("Get ready to speak...")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

# Beep sound
winsound.Beep(1000, 500)

print(f"🎙 Recording for {DURATION} seconds...")
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

print("✅ Recording complete! Processing...")

# Save temporary WAV file
temp_file = "temp_audio.wav"
sf.write(temp_file, audio_data, SAMPLE_RATE)  # <-- fixed

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

features = extract_features(temp_file).reshape(1, -1)

# Predict
prediction = model.predict(features)
print(f"🎯 Predicted Emotion: {prediction[0]}")

# Cleanup
os.remove(temp_file)
