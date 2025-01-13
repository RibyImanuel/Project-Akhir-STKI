import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
from pydub import AudioSegment
import librosa
import numpy as np
from googletrans import Translator
from tensorflow import keras
import tensorflow as tf

# Set the directory where the dataset is stored
data_dir = "dataset"  # Folder dataset setelah diekstrak

# Ambil daftar subfolder di dalam folder dataset
commands = np.array(tf.io.gfile.listdir(data_dir))

# Hapus folder '_background_noise_' dari daftar commands
commands = commands[commands != "_background_noise_"]

# Konversi ke format list jika diperlukan
commands = list(commands)

# Debug: Tampilkan daftar perintah yang ditemukan
print("Commands:", commands)

# Model & Commands
model = keras.models.load_model("speech_command_model.keras")

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Preprocess Audio Function
def preprocess_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    log_mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)
    log_mel_spec = librosa.power_to_db(log_mel_spec, ref=np.max)
    return log_mel_spec.T


# Predict Command
def predict_command(file_path):
    log_mel_spec = preprocess_audio(file_path)
    log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # Add batch dimension
    prediction = model.predict(log_mel_spec)
    return commands[np.argmax(prediction)]


# Extract Audio from Video
def extract_audio(video_file, audio_file, sample_rate=16000):
    # Load video
    video = VideoFileClip(video_file)
    audio = video.audio
    temp_audio_file = "temp_audio.wav"
    audio.write_audiofile(temp_audio_file)
    audio.close()
    video.close()

    # Use pydub to change sample rate
    audio_segment = AudioSegment.from_wav(temp_audio_file)
    audio_segment = audio_segment.set_frame_rate(sample_rate)
    audio_segment.export(audio_file, format="wav")


# Detect Words in Sentence
def detect_words_in_sentence(audio_path, window_size=1.0, stride=1.0):
    audio, sr = librosa.load(audio_path, sr=16000)
    window_samples = int(window_size * sr)
    stride_samples = int(stride * sr)

    detected_words = []

    for start in range(0, len(audio) - window_samples + 1, stride_samples):
        end = start + window_samples
        segment = audio[start:end]
        segment = librosa.util.fix_length(segment, size=window_samples)

        log_mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=16000, n_mels=64, fmax=8000
        )
        log_mel_spec = librosa.power_to_db(log_mel_spec, ref=np.max).T
        log_mel_spec = np.expand_dims(log_mel_spec, axis=0)

        prediction = model.predict(log_mel_spec)
        predicted_label = commands[np.argmax(prediction)]

        if predicted_label != "silence":
            detected_words.append(predicted_label)

    return " ".join(detected_words)


# Flask Routes
@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        video = request.files["video"]
        language = request.form["language"]

        if video:
            video_path = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(video.filename)
            )
            video.save(video_path)

            # Extract audio and detect commands
            audio_path = video_path.replace(".mp4", ".wav")
            extract_audio(video_path, audio_path)
            detected_sentence = detect_words_in_sentence(audio_path)

            # Translate sentence to chosen language
            translator = Translator()
            translation = translator.translate(
                detected_sentence, src="en", dest=language
            ).text

            return render_template(
                "result.html",
                detected_sentence=detected_sentence,
                translation=translation,
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
