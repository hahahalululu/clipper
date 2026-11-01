import os
import re
import time
import json
import uuid
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import yt_dlp
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Suppress Hugging Face symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Cross-Version MoviePy Compatibility Layer (v1.x and v2.x) ---
try:
    from moviepy import VideoFileClip
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import VideoFileClip
    MOVIEPY_V2 = False

# --- Top 5 Best Video Clipping AI Models (Ordered for Free-Tier Success + Fallback) ---
TOP_5_CLIPPER_MODELS = [
    "gemini-3-flash-preview",      # #1: Best free multimodal model (1,500 RPD free)
    "gemini-2.5-flash",            # #2: Proven multimodal workhorse (1,500 RPD free)
    "gemini-3.1-flash-lite",       # #3: Fastest high-RPM video analyzer (1,000 RPD free)
    "gemini-2.0-flash",            # #4: High-capacity 1M TPM backup (1,500 RPD free)
    "gemini-3.1-pro-preview",      # #5: Flagship Pro brain (Requires Paid Tier; auto-skips on Free Tier)
]

# --- High-Impact Viral Subtitle Color Themes & Font Mappings ---
CAPTION_THEMES = {
    "🔥 Viral Yellow Pop": {
        "highlight": (255, 230, 0),    # Vibrant Electric Yellow (#FFE600)
        "base": (255, 255, 255),        # Clean White (#FFFFFF)
        "stroke": (0, 0, 0),           # Deep Black Outline
    },
    "⚡ Neon Lime Glow": {
        "highlight": (34, 197, 94),     # Neon Lime Green (#22C55E)
        "base": (255, 255, 255),
        "stroke": (0, 0, 0),
    },
    "💎 Electric Cyan": {
        "highlight": (0, 229, 255),     # Radiant Cyan (#00E5FF)
        "base": (255, 255, 255),
        "stroke": (0, 0, 0),
    },
    "🔥 Fire Orange": {
        "highlight": (255, 107, 0),     # Energetic Orange (#FF6B00)
        "base": (255, 255, 255),
        "stroke": (0, 0, 0),
    },
    "⚪ Clean White Minimal": {
        "highlight": (255, 255, 255),
        "base": (210, 210, 210),
        "stroke": (0, 0, 0),
    },
}

FONT_MAP = {
    "Impact (TikTok/Shorts Standard)": r"C:\Windows\Fonts\impact.ttf",
    "Arial Bold": r"C:\Windows\Fonts\arialbd.ttf",
    "Segoe UI Bold": r"C:\Windows\Fonts\segoeuib.ttf",
    "Calibri Bold": r"C:\Windows\Fonts\calibrib.ttf",
}

LANGUAGE_NAMES = {
    "en": "English", "id": "Indonesian", "es": "Spanish", "fr": "French",
    "de": "German", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "ru": "Russian", "hi": "Hindi", "pt": "Portuguese",
    "it": "Italian", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "tl": "Tagalog / Filipino",
    "ms": "Malay", "jw": "Javanese", "su": "Sundanese"
}

# In-memory cache for Whisper models to prevent re-instantiation per clip
_WHISPER_CACHE = {}

def get_whisper_model(model_name: str = "base"):
    """Loads and caches the faster-whisper speech-to-text model on CPU with INT8 quantization."""
    if model_name not in _WHISPER_CACHE:
        from faster_whisper import WhisperModel
        _WHISPER_CACHE[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _WHISPER_CACHE[model_name]

# --- Auto-Detect FFmpeg Binary ---
def get_ffmpeg_binary() -> str | None:
    """Finds FFmpeg on system PATH or falls back to the binary bundled with MoviePy/imageio."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

FFMPEG_BINARY = get_ffmpeg_binary()
if FFMPEG_BINARY:
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_BINARY

def clean_ansi(text: str) -> str:
    """Strips ugly terminal color codes from error messages."""
    return re.sub(r'\x1b\[[0-9;]*m', '', str(text))

def format_seconds(seconds: float) -> str:
    """Formats seconds into a clean, readable timer string (e.g., 1m 14.2s or 18.5s)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"

# --- Structured Output Schema for Gemini ---
class HighlightClip(BaseModel):
    title: str = Field(
        description="A high-curiosity, punchy viral title in American English under 60 characters with 1-2 emojis for YouTube Shorts/TikTok"
    )
    caption: str = Field(
        description="Engaging 2-sentence social media caption in American English with a question at the end to force viewer comments. If original video is in another language, translate it to English!"
    )
    hashtags: str = Field(
        description="8 to 10 space-separated trending hashtags targeting US/Canada viewers (e.g., #fyp #viral #trending #usa)"
    )
    best_post_time_wib: str = Field(
        description="Exact recommended posting time in Jakarta Time (WIB) and the corresponding US/Canada Eastern Time (EST/EDT) based on the clip's mood"
    )
    start_time: float = Field(description="Start time in seconds (e.g., 12.5)")
    end_time: float = Field(description="End time in seconds (e.g., 55.0)")
    reasoning: str = Field(description="Why this segment has high viral potential and a strong 3-second hook (written in American English)")

# --- Page Configuration & UI Styling ---
st.set_page_config(
    page_title="AI Viral Shorts Clipper (US/Canada Edition)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stVideo { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    div[data-testid="stMetricValue"] { font-size: 1.25rem; }
    </style>
""", unsafe_allow_html=True)

if "generated_clips" not in st.session_state:
    st.session_state.generated_clips = []
if "timing_summary" not in st.session_state:
    st.session_state.timing_summary = None

# --- Core Helper Functions ---
def download_youtube_video(url: str, output_dir: str) -> str:
    """Downloads YouTube video using single-file MP4 fallback or bundled FFmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    output_template = os.path.join(output_dir, f"source_{unique_id}.%(ext)s")
    
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'nocolor': True,
    }
    
    if FFMPEG_BINARY:
        ydl_opts['ffmpeg_location'] = FFMPEG_BINARY
        ydl_opts['merge_output_format'] = 'mp4'
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        final_path = f"{base}.mp4"
        if os.path.exists(final_path):
            return final_path
        if os.path.exists(filename):
            return filename
            
        for f in os.listdir(output_dir):
            if f.startswith(f"source_{unique_id}"):
                return os.path.join(output_dir, f)
                
        raise FileNotFoundError("Video download completed, but the output file could not be found.")

def analyze_video_with_gemini(
    api_key: str,
    video_path: str,
    preferred_model: str,
    num_clips: int,
    timer_placeholder,
    fallback_log_placeholder
) -> tuple[list, str]:
    """Uploads video once and cascades through the Top 5 AI Models if rate-limited (429) or unavailable (404)."""
    client = genai.Client(api_key=api_key)
    video_file = None
    step_start = time.perf_counter()
    
    try:
        timer_placeholder.info("⏱️ Uploading video to Google AI servers... (0.0s)")
        video_file = client.files.upload(file=video_path)
        
        max_wait_seconds = 300
        while video_file.state.name == "PROCESSING":
            elapsed = time.perf_counter() - step_start
            if elapsed >= max_wait_seconds:
                raise TimeoutError("Video processing timed out on Gemini servers.")
            timer_placeholder.info(f"⏱️ Google AI is preparing your video stream... ({format_seconds(elapsed)} elapsed)")
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
            
        if video_file.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini File API processing failed with state: {video_file.state.name}")

        prompt = f"""
        You are an elite viral short-form video producer and social media strategist targeting audiences in the United States and Canada.
        Watch this video carefully and identify the {num_clips} most engaging, standalone segments.
        
        Strict Rules:
        1. Each clip MUST be between 20 and 60 seconds long.
        2. Each clip must start with an irresistible 3-second verbal or visual hook and end on a complete thought.
        3. Write a punchy, curiosity-inducing `title` in American English (under 60 characters) with 1-2 emojis.
        4. Write an engaging 2-sentence social media `caption` in American English that hooks the reader and asks a question at the end to spark comments.
        5. LANGUAGE TRANSLATION RULE: If the original video contains speech or text in ANY language other than English (e.g., Indonesian, Spanish, French, Japanese, etc.), you MUST translate everything and write the `title`, `caption`, and `reasoning` in fluent, captivating American English for US/Canada viewers!
        6. Provide 8 to 10 space-separated `hashtags` mixing broad US/Canada algorithm tags (#fyp #viral #shorts #usa) and niche topic tags.
        7. For `best_post_time_wib`, calculate the exact time the creator in Jakarta, Indonesia (WIB, UTC+7) should post so it hits peak US/Canada Eastern Time (EST/EDT) viewer activity based on the clip's vibe:
           - Educational/Motivational -> Post at 07:00 PM - 08:30 PM WIB (Hits 7:00 AM - 8:30 AM EST Morning Commute)
           - Entertainment/Storytelling/Funny -> Post at 06:30 AM - 08:30 AM WIB (Hits 6:30 PM - 8:30 PM EST Evening Prime Time)
           - Quick Tips/News -> Post at 11:30 PM WIB or Schedule for 12:30 AM WIB (Hits 12:30 PM EST Lunch Break)
        """

        # Build the 5-model priority queue starting with the user's selected model
        model_queue = [preferred_model] + [m for m in TOP_5_CLIPPER_MODELS if m != preferred_model]
        
        response = None
        used_model = preferred_model
        failed_attempts = []

        for idx, candidate_model in enumerate(model_queue, start=1):
            try:
                elapsed = time.perf_counter() - step_start
                timer_placeholder.info(
                    f"🧠 **[Attempt {idx}/{len(model_queue)}]** `{candidate_model}` is analyzing video & writing viral copy... ({format_seconds(elapsed)} elapsed)"
                )
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=[video_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[HighlightClip],
                        temperature=0.3,
                    ),
                )
                used_model = candidate_model
                break
                
            except Exception as err:
                err_str = str(err)
                # Catch Quota Exceeded (429), Not Found (404), Permission/Tier block (403), or Server Overload (503)
                is_recoverable = any(
                    token in err_str
                    for token in ["429", "RESOURCE_EXHAUSTED", "quota", "404", "NOT_FOUND", "no longer available", "503", "UNAVAILABLE", "403"]
                )
                if is_recoverable:
                    reason = "Quota/Free-Tier Limit Reached (429)" if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else "Model Unavailable (404)"
                    failed_attempts.append(f"`{candidate_model}` ({reason})")
                    fallback_log_placeholder.warning(
                        f"🔄 Skipped {', '.join(failed_attempts)} → Automatically switching to next best AI model..."
                    )
                    time.sleep(1)
                    continue
                else:
                    raise err

        if response is None:
            raise RuntimeError(
                f"All 5 AI models hit their limit or were unavailable on your account. Tried: {', '.join(failed_attempts)}."
            )

        return json.loads(response.text), used_model
        
    finally:
        if video_file:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass

def transcribe_segment_with_whisper(
    video_path: str,
    start: float,
    end: float,
    model_name: str = "base",
    uppercase: bool = True
) -> list[dict]:
    """
    Extracts 16kHz mono audio directly via FFmpeg in memory and runs faster-whisper
    with word-level timestamps (word_timestamps=True, vad_filter=True).
    Returns a list of dicts: [{'word': str, 'start': float, 'end': float}]
    All word timestamps are relative to 0.0s (start of the subclip).
    """
    if not FFMPEG_BINARY:
        return []
    
    cmd = [
        FFMPEG_BINARY,
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "-"
    ]
    
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        if not res.stdout:
            return []
        audio_arr = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_arr) < 8000:  # less than 0.5s of audio
            return []
        
        model = get_whisper_model(model_name)
        segments, _ = model.transcribe(audio_arr, word_timestamps=True, vad_filter=True)
        
        raw_words = []
        for s in segments:
            if s.words:
                for w in s.words:
                    word_clean = w.word.strip()
                    if word_clean:
                        raw_words.append({
                            "word": word_clean.upper() if uppercase else word_clean,
                            "start": float(w.start),
                            "end": float(w.end)
                        })
        return raw_words
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return []

def group_words_into_phrases(raw_words: list[dict], max_words: int = 4) -> list[dict]:
    """
    Groups words into short, high-retention phrases (3-5 words) suitable for vertical phone screens.
    Breaks on max word count, natural punctuation (. ? ! ,), or long speech pauses.
    """
    if not raw_words:
        return []

    phrases = []
    current_chunk = []
    
    for i, w in enumerate(raw_words):
        current_chunk.append(w)
        
        # Check natural break conditions
        is_max_len = len(current_chunk) >= max_words
        has_punctuation = w["word"].endswith((".", "!", "?", ","))
        
        has_long_pause = False
        if i + 1 < len(raw_words):
            gap = raw_words[i + 1]["start"] - w["end"]
            if gap > 0.5:
                has_long_pause = True
                
        if is_max_len or has_punctuation or has_long_pause:
            next_start = raw_words[i + 1]["start"] if (i + 1 < len(raw_words)) else (w["end"] + 0.3)
            phrase_end = min(w["end"] + 0.25, next_start)
            
            phrases.append({
                "start": current_chunk[0]["start"],
                "end": max(phrase_end, current_chunk[0]["start"] + 0.3),
                "words": list(current_chunk)
            })
            current_chunk = []
            
    if current_chunk:
        phrases.append({
            "start": current_chunk[0]["start"],
            "end": current_chunk[-1]["end"] + 0.25,
            "words": list(current_chunk)
        })
        
    return phrases

def get_pillow_font(font_name: str, size: int):
    """Loads system truetype font with automatic fallback."""
    font_path = FONT_MAP.get(font_name, "")
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    for fallback in [r"C:\Windows\Fonts\impact.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]:
        if os.path.exists(fallback):
            try:
                return ImageFont.truetype(fallback, size)
            except Exception:
                pass
    return ImageFont.load_default()

def layout_caption_lines(draw, words: list[dict], font, max_width: float) -> list[tuple]:
    """Wraps words into lines so each line stays within safe screen margins."""
    lines = []
    current_line = []
    current_width = 0.0
    space_w = draw.textlength(" ", font=font)

    for item in words:
        w_text = item["word"]
        w_w = draw.textlength(w_text, font=font)
        needed = w_w if not current_line else (current_width + space_w + w_w)
        if needed <= max_width or not current_line:
            current_line.append((item, w_w))
            current_width = needed
        else:
            lines.append((current_line, current_width))
            current_line = [(item, w_w)]
            current_width = w_w
    if current_line:
        lines.append((current_line, current_width))
    return lines

def build_caption_stamper(
    phrases: list[dict],
    video_w: int,
    video_h: int,
    font_name: str = "Impact (TikTok/Shorts Standard)",
    theme_name: str = "🔥 Viral Yellow Pop",
    y_percent: float = 0.72
):
    """
    Returns a MoviePy frame transformation function (gf, t) -> frame
    that dynamically highlights the active spoken word and renders bold drop-shadow subtitles.
    """
    font_size = max(22, int(video_w * 0.075))
    font = get_pillow_font(font_name, font_size)
    theme = CAPTION_THEMES.get(theme_name, CAPTION_THEMES["🔥 Viral Yellow Pop"])
    highlight_color = theme["highlight"]
    base_color = theme["base"]
    stroke_color = theme["stroke"]
    stroke_w = max(2, int(font_size * 0.1))
    line_spacing = int(font_size * 1.25)
    max_line_width = int(video_w * 0.85)

    def stamp_caption(gf, t: float):
        frame = gf(t)
        active_phrase = None
        for p in phrases:
            if p["start"] <= t <= p["end"]:
                active_phrase = p
                break
        if not active_phrase:
            return frame

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        space_w = draw.textlength(" ", font=font)

        lines = layout_caption_lines(draw, active_phrase["words"], font, max_line_width)
        total_block_h = len(lines) * line_spacing
        start_y = int((video_h * y_percent) - (total_block_h / 2.0))

        for l_idx, (line_words, line_width) in enumerate(lines):
            line_y = start_y + (l_idx * line_spacing)
            curr_x = (video_w - line_width) / 2.0

            for item, w_w in line_words:
                is_active = (item["start"] <= t <= item["end"])
                fill_color = highlight_color if is_active else base_color

                # 1. High-contrast Drop Shadow (2px offset)
                draw.text(
                    (curr_x + 2, line_y + 2),
                    item["word"],
                    font=font,
                    fill=(0, 0, 0),
                    stroke_width=stroke_w,
                    stroke_fill=(0, 0, 0)
                )

                # 2. Main Text with Crisp Black Stroke
                draw.text(
                    (curr_x, line_y),
                    item["word"],
                    font=font,
                    fill=fill_color,
                    stroke_width=stroke_w,
                    stroke_fill=stroke_color
                )
                curr_x += w_w + space_w

        return np.asarray(img)

    return stamp_caption

def render_vertical_clips(
    video_path: str,
    highlights: list,
    output_dir: str,
    progress_bar,
    render_timer_box,
    burn_subtitles: bool = True,
    caption_config: dict = None
) -> list:
    """Cuts video into subclips, center-crops to 9:16, transcribes speech with Whisper, stamps scrolling captions frame-by-frame, and renders MP4s."""
    os.makedirs(output_dir, exist_ok=True)
    rendered_results = []
    video = VideoFileClip(video_path)
    
    if caption_config is None:
        caption_config = {}
    
    try:
        width, height = video.size
        duration = video.duration
        target_ratio = 9.0 / 16.0
        current_ratio = width / height

        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / target_ratio)

        # Ensure dimensions are even integers for libx264 compatibility
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        x_center, y_center = width / 2.0, height / 2.0
        x1 = int(x_center - (new_width / 2.0))
        x2 = x1 + new_width
        y1 = int(y_center - (new_height / 2.0))
        y2 = y1 + new_height

        total_clips = len(highlights)
        for idx, item in enumerate(highlights):
            clip_start_timer = time.perf_counter()
            start = max(0.0, float(item.get("start_time", 0.0)))
            end = min(float(duration), float(item.get("end_time", duration)))
            
            if end <= start:
                continue
            if (end - start) < 5.0:
                end = min(float(duration), start + 15.0)

            render_timer_box.info(f"✂️ Cutting Short #{idx + 1} of {total_clips} ({start:.1f}s - {end:.1f}s)...")

            if MOVIEPY_V2:
                sub = video.subclipped(start, end)
                cropped = sub.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
            else:
                sub = video.subclip(start, end)
                cropped = sub.crop(x1=x1, y1=y1, x2=x2, y2=y2)

            words_transcribed = 0
            final_clip = cropped

            if burn_subtitles:
                try:
                    render_timer_box.info(f"🎙️ Transcribing audio with Whisper STT for Short #{idx + 1}...")
                    words = transcribe_segment_with_whisper(
                        video_path=video_path,
                        start=start,
                        end=end,
                        model_name=caption_config.get("whisper_model", "base"),
                        uppercase=caption_config.get("uppercase", True)
                    )
                    words_transcribed = len(words)
                    
                    if words:
                        render_timer_box.info(f"🔥 Burning {words_transcribed} scrolling words onto Short #{idx + 1}...")
                        phrases = group_words_into_phrases(
                            words,
                            max_words=caption_config.get("words_per_phrase", 4)
                        )
                        stamper = build_caption_stamper(
                            phrases=phrases,
                            video_w=new_width,
                            video_h=new_height,
                            font_name=caption_config.get("font_name", "Impact (TikTok/Shorts Standard)"),
                            theme_name=caption_config.get("theme", "🔥 Viral Yellow Pop"),
                            y_percent=caption_config.get("y_percent", 0.72)
                        )
                        if MOVIEPY_V2:
                            final_clip = cropped.transform(stamper)
                        else:
                            final_clip = cropped.fl(stamper)
                    else:
                        render_timer_box.info(f"ℹ️ No speech detected in Short #{idx + 1}; rendering clean video...")
                except Exception as trans_err:
                    render_timer_box.warning(f"⚠️ Subtitles skipped for Short #{idx + 1}: {clean_ansi(trans_err)}")
                    final_clip = cropped

            output_file = os.path.join(output_dir, f"short_{idx + 1}_{uuid.uuid4().hex[:6]}.mp4")

            render_timer_box.info(f"💾 Encoding 9:16 MP4 video for Short #{idx + 1}...")
            if MOVIEPY_V2:
                final_clip.write_videofile(
                    output_file, codec="libx264", audio_codec="aac",
                    temp_audiofile_path=output_dir, remove_temp=True,
                    preset="fast", threads=4, logger=None
                )
            else:
                final_clip.write_videofile(
                    output_file, codec="libx264", audio_codec="aac",
                    temp_audiofile=os.path.join(output_dir, f"temp_audio_{idx}.m4a"),
                    remove_temp=True, preset="fast", threads=4, verbose=False, logger=None
                )

            clip_render_seconds = time.perf_counter() - clip_start_timer

            rendered_results.append({
                "title": item.get("title", f"Highlight Clip #{idx + 1}"),
                "caption": item.get("caption", "Watch until the end! What do you think? 👇"),
                "hashtags": item.get("hashtags", "#shorts #viral #fyp #trending #usa"),
                "best_post_time_wib": item.get("best_post_time_wib", "07:00 AM WIB (Hits 7:00 PM EST US Evening Peak)"),
                "start": round(start, 1),
                "end": round(end, 1),
                "duration": round(end - start, 1),
                "render_time": format_seconds(clip_render_seconds),
                "reasoning": item.get("reasoning", ""),
                "file_path": output_file,
                "has_subtitles": (words_transcribed > 0 and burn_subtitles),
                "word_count": words_transcribed
            })
            progress_bar.progress((idx + 1) / total_clips)

    finally:
        video.close()

    return rendered_results

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    env_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        placeholder="AIzaSy...",
        help="Paste your secret key from Google AI Studio."
    )
    
    model_choice = st.selectbox(
        "Primary AI Model (Auto-Fallbacks Enabled)",
        options=TOP_5_CLIPPER_MODELS,
        index=0,
        help="Starts with your chosen model. If it hits a quota limit (429), it automatically switches through the other 4 models!"
    )
    
    with st.expander("🛡️ View 5-Model Auto-Fallback Chain"):
        st.caption(
            "If your chosen model hits a daily limit, the app automatically cascades in this order:\n"
            "1. `gemini-3-flash-preview` *(Best Free)*\n"
            "2. `gemini-2.5-flash` *(Reliable Free)*\n"
            "3. `gemini-3.1-flash-lite` *(Fastest Free)*\n"
            "4. `gemini-2.0-flash` *(1M Token Backup)*\n"
            "5. `gemini-3.1-pro-preview` *(Paid Tier)*"
        )
    
    num_clips = st.slider("Number of Shorts to Extract", min_value=1, max_value=5, value=2)
    
    st.divider()
    st.header("💬 Scrolling Captions (Whisper STT)")
    burn_subtitles = st.checkbox(
        "🔥 Burn Scrolling Captions onto Video",
        value=True,
        help="Transcribes spoken audio frame-by-frame using Whisper STT and permanently burns animated scrolling subtitles onto the video for mute viewers."
    )

    if burn_subtitles:
        caption_theme = st.selectbox(
            "Caption Color Theme",
            options=list(CAPTION_THEMES.keys()),
            index=0,
            help="High-contrast color styling with heavy drop shadows."
        )
        caption_font = st.selectbox(
            "Subtitle Font",
            options=list(FONT_MAP.keys()),
            index=0,
            help="Impact and Arial Bold are the viral industry standard for TikTok, Reels & Shorts."
        )
        whisper_model_choice = st.selectbox(
            "Whisper STT Model",
            options=["base (Recommended - Accurate & Fast)", "tiny (Ultra-Fast)", "small (Maximum Accuracy)"],
            index=0,
            help="'base' yields the most accurate speech timestamps with fast rendering."
        )
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            words_per_phrase = st.slider(
                "Words per Screen",
                min_value=2,
                max_value=6,
                value=4,
                help="Shorter chunks (3-4 words) keep viewers engaged."
            )
        with col_w2:
            caption_y = st.slider(
                "Vertical Pos (%)",
                min_value=50,
                max_value=85,
                value=72,
                help="70-75% keeps captions inside the TikTok/Shorts safe zone."
            )
        uppercase_captions = st.checkbox("ALL CAPS Viral Style", value=True, help="Renders all caption words in bold uppercase.")
    else:
        caption_theme = "🔥 Viral Yellow Pop"
        caption_font = "Impact (TikTok/Shorts Standard)"
        whisper_model_choice = "base"
        words_per_phrase = 4
        caption_y = 72
        uppercase_captions = True

    caption_config = {
        "theme": caption_theme,
        "font_name": caption_font,
        "whisper_model": whisper_model_choice.split()[0],
        "words_per_phrase": words_per_phrase,
        "y_percent": caption_y / 100.0,
        "uppercase": uppercase_captions,
    }

    st.divider()
    if FFMPEG_BINARY:
        st.success("✅ Video Engine (FFmpeg) Ready")
    else:
        st.warning("⚠️ Run `pip install imageio-ffmpeg` in terminal if video cutting fails.")

    st.markdown("### 🌎 US/Canada Schedule Guide")
    st.caption(
        "**Your Morning (06:00 - 09:00 AM WIB):**\n"
        "👉 Hits US/Canada Evening Prime Time (Best overall!)\n\n"
        "**Your Night (07:00 - 09:00 PM WIB):**\n"
        "👉 Hits US/Canada Morning Commute"
    )

# --- Main Dashboard UI ---
st.title("🎬 AI Viral Shorts Clipper & US/Canada Copywriter")
st.markdown("Cut **9:16 vertical videos** and automatically generate **viral titles, captions, tags, and Jakarta (WIB) posting times** with **5-Model Auto-Fallback**.")

url_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
with url_col:
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
with btn_col:
    start_btn = st.button("✨ Generate Shorts", type="primary", use_container_width=True)

# --- Execution Pipeline ---
if start_btn:
    if not api_key_input.strip():
        st.error("⚠️ Please paste your Gemini API Key in the left sidebar first.")
    elif not youtube_url.strip() or ("youtube.com" not in youtube_url and "youtu.be" not in youtube_url):
        st.error("⚠️ Please paste a valid YouTube link.")
    else:
        temp_dir = "temp_workspace"
        output_dir = "rendered_shorts"
        source_video_path = None
        pipeline_start = time.perf_counter()
        
        with st.status("🚀 Working on your viral clips...", expanded=True) as status:
            try:
                # Step 1: Download with Timer
                t1_start = time.perf_counter()
                st.write("📥 **Step 1/3:** Downloading video from YouTube...")
                source_video_path = download_youtube_video(youtube_url, temp_dir)
                t1_elapsed = time.perf_counter() - t1_start
                st.write(f"✅ Video downloaded in **{format_seconds(t1_elapsed)}**.")
                
                # Step 2: Gemini Analysis with 5-Model Auto-Fallback & Live Stopwatch
                t2_start = time.perf_counter()
                st.write(f"🧠 **Step 2/3:** Finding best moments & writing US/Canada copy (Starting with `{model_choice}`)...")
                fallback_log_box = st.empty()
                ai_timer_box = st.empty()
                
                highlights, actual_model = analyze_video_with_gemini(
                    api_key=api_key_input.strip(),
                    video_path=source_video_path,
                    preferred_model=model_choice,
                    num_clips=num_clips,
                    timer_placeholder=ai_timer_box,
                    fallback_log_placeholder=fallback_log_box
                )
                t2_elapsed = time.perf_counter() - t2_start
                ai_timer_box.empty()
                
                if not highlights:
                    raise ValueError("Gemini did not return any valid highlight segments.")
                st.write(f"✅ Created **{len(highlights)}** viral clip packages using **`{actual_model}`** in **{format_seconds(t2_elapsed)}**.")
                
                # Step 3: Crop & Render with Timer
                t3_start = time.perf_counter()
                st.write("✂️ **Step 3/3:** Cropping to 9:16 vertical format, syncing Whisper subtitles & saving MP4 files...")
                render_timer_box = st.empty()
                render_progress = st.progress(0.0)
                st.session_state.generated_clips = render_vertical_clips(
                    video_path=source_video_path,
                    highlights=highlights,
                    output_dir=output_dir,
                    progress_bar=render_progress,
                    render_timer_box=render_timer_box,
                    burn_subtitles=burn_subtitles,
                    caption_config=caption_config
                )
                t3_elapsed = time.perf_counter() - t3_start
                render_timer_box.empty()
                st.write(f"✅ All videos cropped and rendered in **{format_seconds(t3_elapsed)}**.")
                
                total_elapsed = time.perf_counter() - pipeline_start
                st.session_state.timing_summary = {
                    "total": format_seconds(total_elapsed),
                    "download": format_seconds(t1_elapsed),
                    "ai": format_seconds(t2_elapsed),
                    "render": format_seconds(t3_elapsed),
                    "model": actual_model
                }
                
                status.update(
                    label=f"🎉 All viral clips ready in {format_seconds(total_elapsed)} (Powered by {actual_model})!",
                    state="complete",
                    expanded=False
                )
                
            except Exception as e:
                status.update(label="❌ Something went wrong", state="error", expanded=True)
                st.error(f"**Error Details:** {clean_ansi(e)}")
                
            finally:
                if source_video_path and os.path.exists(source_video_path):
                    try:
                        os.remove(source_video_path)
                    except Exception:
                        pass

# --- Results Gallery UI ---
if st.session_state.generated_clips:
    st.divider()
    
    if st.session_state.timing_summary:
        ts = st.session_state.timing_summary
        st.subheader("⏱️ Speed & Time Report")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("🏁 Total Time Taken", ts["total"])
        t_col2.metric("📥 YouTube Download", ts["download"])
        t_col3.metric(f"🧠 AI Analysis ({ts['model']})", ts["ai"])
        t_col4.metric("✂️ Video Cutting", ts["render"])
        st.divider()

    st.subheader(f"📱 Your Ready-to-Post Viral Packages ({len(st.session_state.generated_clips)} Ready)")
    
    cols = st.columns(min(len(st.session_state.generated_clips), 2))
    
    for idx, clip in enumerate(st.session_state.generated_clips):
        col = cols[idx % len(cols)]
        with col:
            with st.container(border=True):
                st.markdown(f"### 🎬 Clip #{idx + 1}: {clip['title']}")
                
                if os.path.exists(clip["file_path"]):
                    st.video(clip["file_path"])
                
                if clip.get("has_subtitles"):
                    st.success(f"🔥 **Burned-In Captions Active:** {clip.get('word_count', 0)} spoken words synced frame-by-frame for mute viewing.")
                else:
                    st.caption("ℹ️ **Captions:** Clean video without burned subtitles.")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Clip Timestamp", f"{clip['start']}s - {clip['end']}s")
                m2.metric("Video Length", f"{clip['duration']}s")
                m3.metric("⏱️ Time to Cut", clip.get("render_time", "N/A"))
                
                st.success(f"🕒 **When to Post (Jakarta WIB):** {clip['best_post_time_wib']}")
                
                st.markdown("**📋 1. Copy-Paste Video Title** *(Click the top-right icon inside the box)*:")
                st.code(clip["title"], language=None)
                
                st.markdown("**📝 2. Copy-Paste Video Caption / Description:**")
                st.code(f"{clip['caption']}\n\n{clip['hashtags']}", language=None)
                
                st.markdown("**🏷️ 3. Copy-Paste Tags Only:**")
                st.code(clip["hashtags"], language=None)
                
                st.info(f"💡 **Why the AI picked this moment:** {clip['reasoning']}")
                
                if os.path.exists(clip["file_path"]):
                    with open(clip["file_path"], "rb") as file_bytes:
                        st.download_button(
                            label="⬇️ Download 9:16 MP4 Video",
                            data=file_bytes,
                            file_name=os.path.basename(clip["file_path"]),
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"dl_{idx}_{clip['file_path']}"
                        )