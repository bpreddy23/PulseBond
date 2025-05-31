# Install necessary packages
!pip install -q torchaudio librosa soundfile resemblyzer

import os
import torch
import torchaudio
import librosa
import numpy as np
import pickle
from resemblyzer import VoiceEncoder, preprocess_wav
import glob
from google.colab import files

# ========== Feature Extraction ==========
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

# ========== Step 1: User Registration ==========
def register_user():
  while True:
    name = input("🧑 Enter your name or ID: ").strip()
    profile_dir = f"profile_data/{name}"
    if os.path.exists(profile_dir):
        print(f"❌ The username '{name}' already exists. Please choose a different one.")
    else:
        break
  print("🎤 Upload 3-5 WAV files each for Neutral, Happy, Sad moods.")
  print("📛 Use the format: Neutral_1.wav, Happy_1.wav, Sad_1.wav ...")

  uploaded = files.upload()

  encoder = VoiceEncoder()
  moods = ["Neutral", "Happy", "Sad"]
  tone_features = {}
  all_neutral_embeddings = []

  for mood in moods:
      files_for_mood = [f for f in uploaded if f.startswith(mood)]
      if not files_for_mood:
          raise FileNotFoundError(f"No files found for mood: {mood}")

      pitch_list, energy_list = [], []

      for file in files_for_mood:
          pitch, energy = extract_pitch_energy(file)
          pitch_list.append(pitch)
          energy_list.append(energy)
          if mood == "Neutral":
              wav = preprocess_wav(file)
              all_neutral_embeddings.append(encoder.embed_utterance(wav))

      tone_features[mood] = {
          "pitch": np.mean(pitch_list),
          "energy": np.mean(energy_list)
      }

  avg_embedding = np.mean(all_neutral_embeddings, axis=0)
  save_profile(name, avg_embedding, tone_features)
  print(f"\n✅ Registered '{name}' with average voice and tone features.")

# ========== Step 2: Mood Detection ==========
def detect_mood(name):
    print("🎧 Upload a new audio sample for mood detection (any .wav format).")
    uploaded = files.upload()

    # Get the filename of the uploaded .wav file
    wav_file = next((f for f in uploaded if f.endswith('.wav')), None)

    if not wav_file:
        print("❌ Please upload a valid .wav file.")
        return

    print(f"📜 Analyzing mood from file: {wav_file}")

    encoder = VoiceEncoder()
    test_wav = preprocess_wav(wav_file)
    test_embed = encoder.embed_utterance(test_wav)
    ref_embed, tone_data = load_profile(name)

    # Speaker Verification
    similarity = np.dot(test_embed, ref_embed) / (np.linalg.norm(test_embed) * np.linalg.norm(ref_embed))
    print(f"🔍 Speaker similarity score: {similarity:.2f}")
    if similarity < 0.84:
        print("❌ Voice does not match the registered user.")
        return

    # Mood Detection
    pitch, energy = extract_pitch_energy(wav_file)
    distances = {}
    for mood, features in tone_data.items():
        mp, me = features["pitch"], features["energy"]
        dist = np.sqrt((pitch - mp)**2 + (energy - me)**2)
        distances[mood] = dist

    mood_detected = min(distances, key=distances.get)
    print(f"✅ Mood Detected: {mood_detected}")

# ========== RUN ==========
print("Choose an option:")
print("1. Register a new user")
print("2. Detect mood from test voice")

# Infinite loop for continuous testing
while True:
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        register_user()
    elif choice == "2":
        name = input("🔐 Enter your registered name/ID: ").strip()
        detect_mood(name)
    else:
        print("❌ Invalid choice.")

    # Option to continue testing or exit
    continue_choice = input("\nWould you like to test another voice? (yes/no): ").strip().lower()
    if continue_choice != 'yes':
        break
