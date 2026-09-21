import librosa
import numpy as np


def extract_features(audio_path):
    """
    Extracts MFCC + Chroma + Mel Spectrogram features from an audio file.
    Returns a 1D combined feature array suitable for the LSTM model.
    """
    try:
        y, sr = librosa.load(audio_path, duration=3, offset=0.5, res_type="kaiser_fast")

        # MFCCs — 40 coefficients, mean across time
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)  # shape: (40,)

        # Chroma — 12 pitch classes, mean across time
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma = np.mean(chroma.T, axis=0)  # shape: (12,)

        # Mel Spectrogram — 128 bands, mean across time
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel = np.mean(mel.T, axis=0)  # shape: (128,)

        # Combined: (180,)
        features = np.hstack((mfccs, chroma, mel))
        return features

    except Exception as e:
        print(f"[feature_extraction] Error: {e}")
        return None