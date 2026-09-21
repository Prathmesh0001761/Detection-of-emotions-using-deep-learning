import numpy as np
import tensorflow as tf
from utils.feature_extraction import extract_features

# Load trained model
model = tf.keras.models.load_model("models/emotion_model.h5")

# Define emotions
emotions_dict = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful"
}

def predict_emotion(audio_path):
    """
    Predicts the emotion from an audio file.
    """
    features = extract_features(audio_path)
    if features is None:
        print("Could not process the audio file.")
        return

    # Reshape for model
    features = np.expand_dims(features, axis=0)

    # Predict emotion
    prediction = model.predict(features)
    predicted_emotion = emotions_dict[np.argmax(prediction)]

    print(f"Predicted Emotion: {predicted_emotion}")

# Example usage
if __name__ == "__main__":
    test_audio = "recordings/test_audio.wav"
    predict_emotion(test_audio)
