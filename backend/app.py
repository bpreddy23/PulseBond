from flask import Flask, request, jsonify
import os
import numpy as np
import pickle
from resemblyzer import VoiceEncoder, preprocess_wav
import librosa

app = Flask(__name__)

# Utility functions (same as your code)
def extract_pitch_energy(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0
    energy = np.mean(librosa.feature.rms(y=y))
    return pitch, energy

def save_profile(name, embedding, tone_features):
    profile_dir = f"profile_data/{name}"
    os.makedirs(profile_dir, exist_ok=True)
    with open(f"{profile_dir}/voice_embedding.pkl", "wb") as f:
        pickle.dump(embedding, f)
    with open(f"{profile_dir}/tone_features.pkl", "wb") as f:
        pickle.dump(tone_features, f)

def load_profile(name):
    profile_dir = f"profile_data/{name}"
    with open(f"{profile_dir}/voice_embedding.pkl", "rb") as f:
        embedding = pickle.load(f)
    with open(f"{profile_dir}/tone_features.pkl", "rb") as f:
        tone_features = pickle.load(f)
    return embedding, tone_features

encoder = VoiceEncoder()  # Initialize once to save time

@app.route('/register', methods=['POST'])
def register():
    # Expecting form-data with 'name' and multiple wav files
    name = request.form.get('name')
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    profile_dir = f"profile_data/{name}"
    if os.path.exists(profile_dir):
        return jsonify({'error': 'Username already exists'}), 400

    # Save wav files temporarily
    moods = ["Neutral", "Happy", "Sad"]
    tone_features = {}
    all_neutral_embeddings = []

    os.makedirs(profile_dir, exist_ok=True)

    for mood in moods:
        files_for_mood = [f for f in request.files if f.startswith(mood)]
        if not files_for_mood:
            return jsonify({'error': f"No files found for mood: {mood}"}), 400

        pitch_list, energy_list = [], []

        for filename in request.files:
            if filename.startswith(mood):
                wav_file = request.files[filename]
                filepath = os.path.join(profile_dir, filename)
                wav_file.save(filepath)

                pitch, energy = extract_pitch_energy(filepath)
                pitch_list.append(pitch)
                energy_list.append(energy)

                if mood == "Neutral":
                    wav = preprocess_wav(filepath)
                    all_neutral_embeddings.append(encoder.embed_utterance(wav))

        tone_features[mood] = {
            "pitch": np.mean(pitch_list),
            "energy": np.mean(energy_list)
        }

    avg_embedding = np.mean(all_neutral_embeddings, axis=0)
    save_profile(name, avg_embedding, tone_features)

    return jsonify({'message': f"User {name} registered successfully."})

@app.route('/detect', methods=['POST'])
def detect():
    name = request.form.get('name')
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'Audio file is required'}), 400

    audio_file = request.files['file']

    profile_dir = f"profile_data/{name}"
    if not os.path.exists(profile_dir):
        return jsonify({'error': 'User not found'}), 404

    filepath = os.path.join(profile_dir, 'temp.wav')
    audio_file.save(filepath)

    test_wav = preprocess_wav(filepath)
    test_embed = encoder.embed_utterance(test_wav)
    ref_embed, tone_data = load_profile(name)

    similarity = np.dot(test_embed, ref_embed) / (np.linalg.norm(test_embed) * np.linalg.norm(ref_embed))
    if similarity < 0.84:
        return jsonify({'error': 'Voice does not match registered user'}), 403

    pitch, energy = extract_pitch_energy(filepath)
    distances = {}
    for mood, features in tone_data.items():
        mp, me = features["pitch"], features["energy"]
        dist = np.sqrt((pitch - mp)**2 + (energy - me)**2)
        distances[mood] = dist

    mood_detected = min(distances, key=distances.get)

    return jsonify({'mood': mood_detected})

if __name__ == '__main__':
    app.run(debug=True)
