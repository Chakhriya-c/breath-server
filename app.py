from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import os
import logging

app = Flask(__name__)
CORS(app)  # To handle CORS issues, if needed

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model
try:
    model = tf.keras.models.load_model('model.h5')
    logger.info('Model loaded successfully.')
except Exception as e:
    logger.error(f'Error loading model: {e}')

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def segment_audio(y, sr, segment_length=2, hop_length=1):
    frames = []
    for start in range(0, len(y) - int(segment_length * sr), int(hop_length * sr)):
        end = start + int(segment_length * sr)
        segment = y[start:end]
        frames.append(extract_features(segment, sr))
    return np.array(frames)

def predict_periods(model, y, sr, segment_length=2, hop_length=1):
    frames = segment_audio(y, sr, segment_length, hop_length)
    predictions = model.predict(frames)
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

@app.route('/predict', methods=['POST'])
def predict():
    logger.info('Prediction endpoint called.')
    if 'audio' not in request.files:
        logger.error('No audio file provided')
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    logger.info(f'File saved at {file_path}')

    try:
        y, sr = process_audio(file_path)
        predicted_labels = predict_periods(model, y, sr)
        periods, durations = get_periods_and_durations(predicted_labels)

        # Clean up file
        os.remove(file_path)
        logger.info(f'File removed from {file_path}')

        # Convert periods and durations to readable format
        results = []
        for i, (period, duration) in enumerate(zip(periods, durations)):
            label = 'Inhale' if period == 0 else 'Exhale'
            results.append({'period': i + 1, 'label': label, 'duration': duration})

        return jsonify({'results': results})
    except Exception as e:
        logger.error(f'Error during prediction: {e}')
        os.remove(file_path)  # Clean up file in case of error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
