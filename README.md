# Call Center Audio Analyzer (Local) using AI/NLP/speech models

This Python script processes a folder of `.wav` call recordings locally and produces statistics including:

- Call duration  
- Spam call detection  
- Good call detection (basic heuristic)  
- Sentiment analysis of call text  
- Summary statistics like average call duration  

---

## Features

- Uses **OpenAI Whisper** for local speech-to-text transcription (English)  
- Uses **HuggingFace DistilBERT sentiment model** for local sentiment classification  
- Spam and good call detection via simple keyword matching (customizable)  
- Runs fully locally, no internet or cloud dependencies once models are downloaded  

---

## Requirements

- Python 3.8+  
- FFmpeg installed and in system PATH (required by pydub)  
- Python packages: `pydub`, `whisper`, `transformers`, `torch`  

---

## Installation

1. Install FFmpeg:

- **Ubuntu/Debian:**  
```
sudo apt update
sudo apt install ffmpeg
```

- **Windows:** 
Download from https://ffmpeg.org/download.html and add to PATH.

2. Create and activate a Python virtual environment (optional but recommended)

```
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install Python dependencies:
```
python analyze_calls_en.py
```

4. The script will process each file, output basic info per file, and print a summary at the end.

## Notes

- Whisper model will download the first time it runs (~70-150MB depending on model size).

- Sentiment model will also be downloaded on first run.

- Modify spam keywords and good call phrases in the script as needed to fit your use case.

- For better spam/good call detection, consider collecting labeled data and training a custom classifier.

## Extending
- Add speaker diarization to separate customer/agent voices (look into pyannote.audio)

- Add topic modeling or advanced NLP for better behavior analytics

- Integrate with a database or dashboard for monitoring call center quality metrics

## Contacts
To contact me please pm:

Telegram: https://t.me/ryu8777

Discord: https://discord.gg/zSw58e9Uvf

