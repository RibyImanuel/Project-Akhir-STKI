from flask import Flask, request, render_template
from googletrans import Translator
import os
from moviepy import VideoFileClip
import speech_recognition as sr

app = Flask(__name__)

# Fungsi untuk mengekstrak audio dari video
def extract_audio(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)
    audio.close()
    video.close()

# Fungsi untuk mentranskripsi audio menjadi teks
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        detected_text = recognizer.recognize_google(audio, language="id-ID")
        return detected_text
    except Exception as e:
        return str(e)

# Fungsi untuk menerjemahkan teks
def translate_text(text, target_language='en'):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        return str(e)

@app.route("/", methods=["GET", "POST"])
def index():
    detected_text = None
    translated_text = None
    
    if request.method == "POST":
        video_file = request.files["video"]
        language = request.form["language"]

        # Menyimpan video sementara untuk diproses
        video_path = os.path.join("uploads", video_file.filename)
        video_file.save(video_path)

        # Menyimpan audio sebagai WAV
        audio_path = os.path.join("uploads", "audio.wav")
        extract_audio(video_path, audio_path)

        # Menyaring audio dan mengonversinya ke teks
        detected_text = transcribe_audio(audio_path)

        # Menerjemahkan teks
        if detected_text:
            translated_text = translate_text(detected_text, target_language=language)

    return render_template("index.html", detected_text=detected_text, translated_text=translated_text)

if __name__ == "__main__":
    app.run(debug=True)
