import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Path to dataset
DATA_PATH = os.path.join("data", "ravdess_speech")
def get_emotion_label(file_path):
    # Extract emotion code from filename
    emotion_code = int(os.path.basename(file_path).split("-")[2])
    emotion_map = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotion_map.get(emotion_code, "unknown")

# Extract features from audio
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

# Load dataset
import glob
import numpy as np

def load_data(test_size=0.2):
    X, Y = [], []

    # Recursively search all wav files inside data/ravdess_speech
    wav_files = glob.glob("data/ravdess_speech/**/*.wav", recursive=True)

    if len(wav_files) == 0:
        raise FileNotFoundError("No audio files found. Check your data path.")

    for file in wav_files:
        try:
            feature = extract_feature(file)
            X.append(feature)
            Y.append(get_emotion_label(file))
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    return train_test_split(np.array(X), Y, test_size=test_size, random_state=42)


# Train model
def train_model():
    x_train, x_test, y_train, y_test = load_data()
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Save model
    with open("audio_emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as audio_emotion_model.pkl")

if __name__ == "__main__":
    train_model()

