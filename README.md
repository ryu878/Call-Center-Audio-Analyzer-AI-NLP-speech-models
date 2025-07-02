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
  ```bash
  sudo apt update
  sudo apt install ffmpeg```

- **Windows:** 
Download from https://ffmpeg.org/download.html and add to PATH.

- **Create and activate a Python virtual environment (optional but recommended):**  

```python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows```
