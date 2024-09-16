from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import os
import shutil

app = FastAPI()

# Allow CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the .h5 model
model = tf.keras.models.load_model('model.h5')

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def segment_audio(y, sr, segment_length=2, hop_length=1):
    frames = []
    for start in range(0, len(y) - int(segment_length * sr), int(hop_length * sr)):
        end = start + int(segment_length * sr)
        segment = y[start:end]
        frames.append(extract_features(segment, sr))
    return np.array(frames, dtype=np.float32)

def predict_periods(model, y, sr, segment_length=2, hop_length=1):
    frames = segment_audio(y, sr, segment_length, hop_length)
    
    # Get predictions for each frame
    predictions = model.predict(frames)
    
    # Get the predicted labels (assuming the model outputs probabilities, use argmax)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

def get_periods_and_durations(predicted_labels, segment_length=2, hop_length=1):
    periods = []
    durations = []
    current_label = None
    current_duration = 0

    for i, label in enumerate(predicted_labels):
        if label == current_label:
            current_duration += hop_length
        else:
            if current_label is not None:
                periods.append(current_label)
                durations.append(current_duration)
            current_label = label
            current_duration = hop_length

    if current_label is not None:
        periods.append(current_label)
        durations.append(current_duration)

    return periods, durations

def process_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr

@app.post("/predict/")
async def predict(audio: UploadFile = File(...)):
    # Check for audio file
    if not audio.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav files are accepted.")

    file_path = f"uploads/{audio.filename}"

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        # Process the audio file
        y, sr = process_audio(file_path)
        predicted_labels = predict_periods(model, y, sr)
        periods, durations = get_periods_and_durations(predicted_labels)

        # Convert periods and durations to readable format
        results = []
        for i, (period, duration) in enumerate(zip(periods, durations)):
            label = 'Inhale' if period == 0 else 'Exhale'
            results.append({'period': i + 1, 'label': label, 'duration': duration})

        # Clean up file
        os.remove(file_path)

        return {"results": results}

    except Exception as e:
        os.remove(file_path)  # Clean up file in case of error
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
