# AI Video Clipper & Shorts Generator

An AI-powered video clipping application built with Streamlit, Google Gemini models, and Faster-Whisper. It automatically extracts viral clips from long-form YouTube videos or uploaded files and formats them into high-engagement vertical Shorts/Reels/TikToks with animated captions.

## Features
- **Multimodal AI Analysis**: Powered by Google Gemini (Gemini 2.5 Flash, Gemini 3 Flash Preview, etc.) to detect highlight-worthy segments.
- **YouTube & Local Video Support**: Download videos directly via `yt-dlp` or upload your own files.
- **Word-Level Subtitles**: Fast transcription and word-level timestamping using `faster-whisper`.
- **Dynamic Captions**: Customizable font, highlight colors, and outline strokes optimized for vertical video formats (9:16).
- **Automated Cropping & Rendering**: Cropped vertical framing with smooth video rendering via `moviepy` and `ffmpeg`.

## Prerequisites
- Python 3.10+
- FFmpeg installed and available on system PATH (or via `imageio-ffmpeg`)
- Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hahahalululu/clipper.git
   cd clipper
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, enter your Gemini API key, input a video, and generate your viral clips!

