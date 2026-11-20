
import os
import math
import subprocess
from pathlib import Path

import yt_dlp


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helper: Run FFmpeg commands
# ---------------------------------------------------------

def run_command(command: list) -> None:
    """Execute FFmpeg/FFprobe command and report errors."""

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{result.stderr}"
        )


# ---------------------------------------------------------
# Download YouTube Audio
# ---------------------------------------------------------

def download_youtube_audio(url: str) -> str:

    output_path = str(
        DOWNLOAD_DIR / "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "outtmpl_na_placeholder": "unknown",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            filename = ydl.prepare_filename(info)

        wav_path = str(
            Path(filename).with_suffix(".wav")
        )

        if not os.path.exists(wav_path):
            raise FileNotFoundError(
                f"Downloaded WAV file not found: {wav_path}"
            )

        return wav_path

    except Exception as e:
        raise RuntimeError(
            f"YouTube audio download failed: {e}"
        ) from e


# ---------------------------------------------------------
# Convert Local Audio/Video to WAV
# ---------------------------------------------------------

def convert_to_wav(input_path: str) -> str:

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path = str(
        Path(input_path).with_name(
            Path(input_path).stem + "_converted.wav"
        )
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        output_path,
    ]

    run_command(command)

    if not os.path.exists(output_path):
        raise RuntimeError("WAV conversion failed.")

    return output_path


# ---------------------------------------------------------
# Get Audio Duration
# ---------------------------------------------------------

def get_audio_duration(wav_path: str) -> float:

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFprobe failed: {result.stderr}"
        )

    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise RuntimeError(
            "Could not determine audio duration."
        ) from e


# ---------------------------------------------------------
# Split Audio into Chunks
# ---------------------------------------------------------

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(
            f"Audio file not found: {wav_path}"
        )

    if chunk_minutes <= 0:
        raise ValueError(
            "chunk_minutes must be greater than zero."
        )

    print("Checking audio duration...")

    duration = get_audio_duration(wav_path)

    if duration <= 0:
        raise ValueError("Audio file is empty.")

    chunk_seconds = chunk_minutes * 60

    total_chunks = math.ceil(
        duration / chunk_seconds
    )

    chunks = []

    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Total chunks: {total_chunks}")

    base_path = Path(wav_path)

    for i in range(total_chunks):

        start_time = i * chunk_seconds

        output_chunk = str(
            base_path.with_name(
                f"{base_path.stem}_chunk_{i}.wav"
            )
        )

        command = [
            "ffmpeg",
            "-y",
            "-i", wav_path,
            "-ss", str(start_time),
            "-t", str(chunk_seconds),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_chunk,
        ]

        print(
            f"Creating chunk {i + 1}/{total_chunks}..."
        )

        run_command(command)

        if os.path.isfile(output_chunk):
            chunks.append(output_chunk)
        else:
            raise RuntimeError(
                f"Failed to create chunk {i + 1}"
            )

    print("Audio chunking completed.")

    return chunks


# ---------------------------------------------------------
# Main Input Processor
# ---------------------------------------------------------

def process_input(source: str) -> list:

    source = source.strip()

    if not source:
        raise ValueError("Please provide a valid input.")

    if source.startswith(("http://", "https://")):

        print("Detected YouTube URL.")
        print("Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:

        print("Detected local audio/video file.")
        print("Converting to WAV...")

        wav_path = convert_to_wav(source)

    print("Starting audio chunking...")

    chunks = chunk_audio(
        wav_path,
        chunk_minutes=10
    )

    if not chunks:
        raise RuntimeError(
            "No audio chunks were generated."
        )

    print(
        f"Audio ready: {len(chunks)} chunk(s) created."
    )

    return chunks
