import os
from pydub import AudioSegment
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Whisper model locally (English)
whisper_model = whisper.load_model("small")  # small for balance speed/accuracy

# Load English sentiment model locally
# Using 'distilbert-base-uncased-finetuned-sst-2-english' from HuggingFace (binary sentiment)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def get_duration(filepath):
    audio = AudioSegment.from_wav(filepath)
    return audio.duration_seconds

def transcribe(filepath):
    result = whisper_model.transcribe(filepath, language='en', fp16=False)
    return result["text"]

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    # classes: 0 = negative, 1 = positive
    return {"negative": float(probs[0]), "positive": float(probs[1])}

# Basic English keyword spam detection (customize as needed)
SPAM_KEYWORDS_EN = [
    "free money", "win", "credit card", "loan", "subscribe now", "click here", "urgent"
]

def detect_spam(text):
    text = text.lower()
    return any(keyword in text for keyword in SPAM_KEYWORDS_EN)

# Basic English good call phrases
GOOD_PHRASES_EN = [
    "thank you for calling", "how can i help you", "have a nice day", "i understand", "let me assist you"
]

def detect_good_call(text):
    text = text.lower()
    return any(phrase in text for phrase in GOOD_PHRASES_EN)

def analyze_calls(folder_path):
    call_stats = {
        "total_calls": 0,
        "total_duration": 0,
        "spam_calls": 0,
        "good_calls": 0,
        "good_call_duration": 0,
        "calls": []
    }

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            call_stats["total_calls"] += 1

            duration = get_duration(filepath)
            call_stats["total_duration"] += duration

            text = transcribe(filepath)

            is_spam = detect_spam(text)
            if is_spam:
                call_stats["spam_calls"] += 1

            is_good = detect_good_call(text)
            if is_good:
                call_stats["good_calls"] += 1
                call_stats["good_call_duration"] += duration

            sentiment = classify_sentiment(text)

            call_stats["calls"].append({
                "file": filename,
                "duration_sec": duration,
                "spam": is_spam,
                "good": is_good,
                "transcript": text,
                "sentiment": sentiment
            })

            print(f"Processed {filename}: duration={duration:.1f}s, spam={is_spam}, good={is_good}, sentiment={sentiment}")

    call_stats["average_duration"] = call_stats["total_duration"] / call_stats["total_calls"] if call_stats["total_calls"] else 0
    call_stats["average_good_call_duration"] = call_stats["good_call_duration"] / call_stats["good_calls"] if call_stats["good_calls"] else 0

    return call_stats

if __name__ == "__main__":
    folder = "./wav_calls"  # Change this to your folder path
    stats = analyze_calls(folder)
    print("\nSummary:")
    print(f"Total calls: {stats['total_calls']}")
    print(f"Spam calls: {stats['spam_calls']}")
    print(f"Good calls: {stats['good_calls']}")
    print(f"Total duration (s): {stats['total_duration']:.1f}")
    print(f"Average call duration (s): {stats['average_duration']:.1f}")
    print(f"Average good call duration (s): {stats['average_good_call_duration']:.1f}")
