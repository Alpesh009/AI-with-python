import openai
import librosa
import numpy as np

# Set your OpenAI API key here or export it as an environment variable
openai.api_key = ("Add your API key here")  # Replace with your actual key

# === STEP 1: Transcribe Audio ===
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return transcript.text

# === STEP 2: Extract Voice Features ===
def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0

    return {
        "duration_sec": duration,
        "avg_pitch_hz": avg_pitch
    }

# === STEP 3: GPT-4 Confidence Analysis ===
def analyze_confidence(transcript, features):
    prompt = f"""
You are a speech coach. Analyze the speaker's confidence based on their transcript and voice features.

Transcript:
"{transcript}"

Voice Features:
- Average Pitch: {features['avg_pitch_hz']:.2f} Hz
- Duration: {features['duration_sec']:.2f} seconds

Is the speaker confident, nervous, or unsure? Provide reasoning.
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in communication and speech analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# === STEP 4: Filler and Doubtful Word Analysis ===
def analyze_fillers_and_doubtful_words(transcript):
    fillers = ["uh", "um", "like", "you know", "so", "actually", "basically"]
    doubtful = ["maybe", "i think", "i guess", "probably", "not sure", "perhaps", "possibly"]

    words = transcript.lower().split()
    total_words = len(words)

    filler_count = sum(words.count(filler) for filler in fillers)
    doubtful_count = sum(transcript.lower().count(phrase) for phrase in doubtful)

    filler_percent = (filler_count / total_words) * 100 if total_words > 0 else 0
    doubtful_percent = (doubtful_count / total_words) * 100 if total_words > 0 else 0

    # Rough confidence estimate: less fillers/doubt = more confidence
    confidence_score = max(0, 100 - (filler_percent * 2 + doubtful_percent * 3))

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "doubtful_count": doubtful_count,
        "filler_percent": filler_percent,
        "doubtful_percent": doubtful_percent,
        "confidence_score_percent": confidence_score
    }

# === MAIN ===
def main(audio_path):
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)

    print("Extracting voice features...")
    features = extract_voice_features(audio_path)

    print("Analyzing confidence with GPT...")
    analysis = analyze_confidence(transcript, features)

    print("\n===== Transcript =====")
    print(transcript)

    print("\n===== Confidence Analysis (GPT) =====")
    print(analysis)

    print("\n===== Filler & Doubtful Word Analysis =====")
    filler_stats = analyze_fillers_and_doubtful_words(transcript)
    print(f"Total words: {filler_stats['total_words']}")
    print(f"Filler words count: {filler_stats['filler_count']} ({filler_stats['filler_percent']:.2f}%)")
    print(f"Doubtful words count: {filler_stats['doubtful_count']} ({filler_stats['doubtful_percent']:.2f}%)")
    print(f"Estimated confidence score: {filler_stats['confidence_score_percent']:.2f}%")

if __name__ == "__main__":
    audio_file = "sam4.wav"  # Replace with your actual audio file path
    main(audio_file)
