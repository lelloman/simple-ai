#!/usr/bin/env python3
"""SimpleAI Coqui XTTS-v2 provider.

XTTS-v2 is licensed under the Coqui Public Model License and is limited to
non-commercial use of the model and its outputs. Use reference voices only when
you have the rights and consent to clone them.

The Rust runner starts this process with:
  --model <model-id> --port <port>

The process keeps XTTS loaded and exposes:
  GET  /health
  POST /v1/audio/speech

Voice configuration can be provided with --voice-map JSON. Accepted shapes:
  {
    "lello": {"speaker_wav": "/voices/lello.wav", "language": "it"},
    "ana": {"speaker": "Ana Florence", "language": "en"},
    "clone": "/voices/clone.wav"
  }

If --voices-dir is set and a requested voice is not in the map, the provider
looks for <voices-dir>/<voice>.wav and uses it as speaker_wav.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "opus", "aac", "pcm"}
CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}
LANGUAGE_RE = re.compile(r"\blanguage\s*[:=]\s*([a-z]{2}(?:-[a-z]{2})?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class VoiceConfig:
    id: str
    speaker_wav: list[str] | None = None
    speaker: str | None = None
    language: str | None = None


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ffmpeg_exe() -> str | None:
    configured = os.environ.get("SIMPLE_AI_FFMPEG_PATH")
    if configured:
        return configured
    return shutil.which("ffmpeg")


def safe_voice_id(value: str) -> str:
    if not value or any(ch in value for ch in "/\\\0"):
        raise ValueError("voice id must be a simple configured id")
    return value


def normalize_format(value: Any) -> str:
    fmt = str(value or "mp3").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported response_format: {fmt}")
    return fmt


def parse_voice_entry(voice_id: str, value: Any) -> VoiceConfig:
    if isinstance(value, str):
        return VoiceConfig(id=voice_id, speaker_wav=[value])
    if not isinstance(value, dict):
        raise ValueError(f"voice map entry for {voice_id} must be a string or object")

    speaker_wav = value.get("speaker_wav") or value.get("speakerWav")
    if isinstance(speaker_wav, str):
        speaker_wavs = [speaker_wav]
    elif isinstance(speaker_wav, list):
        speaker_wavs = [str(item) for item in speaker_wav]
    elif speaker_wav is None:
        speaker_wavs = None
    else:
        raise ValueError(f"speaker_wav for {voice_id} must be a string or array")

    speaker = value.get("speaker") or value.get("speaker_idx") or value.get("speakerIdx")
    language = value.get("language")
    return VoiceConfig(
        id=voice_id,
        speaker_wav=speaker_wavs,
        speaker=str(speaker) if speaker is not None else None,
        language=str(language) if language is not None else None,
    )


def load_voice_map(path: str | None) -> dict[str, VoiceConfig]:
    if not path:
        return {}
    raw = json.loads(Path(path).expanduser().read_text())
    if not isinstance(raw, dict):
        raise ValueError("voice map must be a JSON object")
    return {safe_voice_id(str(key)): parse_voice_entry(str(key), value) for key, value in raw.items()}


def parse_voice_id(raw_voice: Any) -> str:
    if isinstance(raw_voice, str):
        return safe_voice_id(raw_voice)
    if isinstance(raw_voice, dict) and isinstance(raw_voice.get("id"), str):
        return safe_voice_id(raw_voice["id"])
    raise ValueError("voice must be a string or object with an id field")


def parse_instruction_language(instructions: Any) -> str | None:
    if not isinstance(instructions, str):
        return None
    match = LANGUAGE_RE.search(instructions)
    return match.group(1).lower() if match else None


def ensure_path_exists(paths: list[str]) -> list[str]:
    resolved = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise ValueError(f"speaker_wav does not exist: {path}")
        resolved.append(str(path))
    return resolved


def convert_wav(wav_path: Path, response_format: str) -> tuple[bytes, str]:
    if response_format == "wav":
        return wav_path.read_bytes(), CONTENT_TYPES[response_format]

    ffmpeg = ffmpeg_exe()
    if not ffmpeg:
        raise ValueError(f"ffmpeg is required for response_format={response_format}")

    if response_format == "pcm":
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(wav_path),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-f",
            "s16le",
            "pipe:1",
        ]
        return subprocess.check_output(command), CONTENT_TYPES[response_format]

    suffix = ".m4a" if response_format == "aac" else f".{response_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as converted:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(wav_path),
        ]
        if response_format == "mp3":
            command.extend(["-codec:a", "libmp3lame", "-q:a", "3"])
        elif response_format == "opus":
            command.extend(["-codec:a", "libopus", "-b:a", "64k"])
        elif response_format == "aac":
            command.extend(["-codec:a", "aac", "-b:a", "128k"])
        elif response_format == "flac":
            command.extend(["-codec:a", "flac"])
        command.append(converted.name)
        subprocess.check_call(command)
        return Path(converted.name).read_bytes(), CONTENT_TYPES[response_format]


class XttsProvider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_id = args.model
        self.loaded_at = time.time()
        self.device = choose_device(args.device)
        self.voice_map = load_voice_map(args.voice_map)
        self.voices_dir = Path(args.voices_dir).expanduser() if args.voices_dir else None
        self.lock = threading.Lock()

        try:
            from TTS.api import TTS
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Coqui TTS. Install the maintained package, for example: "
                "pip install coqui-tts soundfile"
            ) from exc

        self.tts = TTS(args.coqui_model_name).to(self.device)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": "coqui-xtts",
            "model": self.model_id,
            "coquiModel": self.args.coqui_model_name,
            "device": self.device,
            "loadedAt": self.loaded_at,
            "voices": sorted(self.available_voices()),
            "defaultLanguage": self.args.default_language,
        }

    def available_voices(self) -> set[str]:
        voices = set(self.voice_map)
        if self.voices_dir and self.voices_dir.exists():
            voices.update(path.stem for path in self.voices_dir.glob("*.wav"))
        return voices

    def voice_config(self, voice_id: str) -> VoiceConfig:
        if voice_id in self.voice_map:
            return self.voice_map[voice_id]
        if self.voices_dir:
            candidate = self.voices_dir / f"{voice_id}.wav"
            if candidate.exists():
                return VoiceConfig(id=voice_id, speaker_wav=[str(candidate)])
        # Let Coqui try built-in speaker ids when no local mapping exists.
        return VoiceConfig(id=voice_id, speaker=voice_id)

    def language_for(self, request: dict[str, Any], voice: VoiceConfig) -> str:
        return str(
            request.get("language")
            or parse_instruction_language(request.get("instructions"))
            or voice.language
            or self.args.default_language
        ).lower()

    def synthesize(self, request: dict[str, Any]) -> tuple[bytes, str]:
        if request.get("model") != self.model_id:
            raise ValueError(
                f"request.model {request.get('model')} does not match loaded model {self.model_id}"
            )
        if request.get("stream_format", "audio") == "sse":
            raise ValueError("stream_format=sse is not implemented by the XTTS provider")

        text = str(request.get("input") or "").strip()
        if not text:
            raise ValueError("input is required")
        voice_id = parse_voice_id(request.get("voice"))
        voice = self.voice_config(voice_id)
        language = self.language_for(request, voice)
        response_format = normalize_format(request.get("response_format"))
        speed = float(request.get("speed") or 1.0)

        with tempfile.TemporaryDirectory(prefix="simple_ai_xtts_") as tmpdir:
            output = Path(tmpdir) / "speech.wav"
            kwargs: dict[str, Any] = {
                "text": text,
                "file_path": str(output),
                "language": language,
                "split_sentences": True,
            }
            if voice.speaker_wav:
                kwargs["speaker_wav"] = ensure_path_exists(voice.speaker_wav)
            elif voice.speaker:
                kwargs["speaker"] = voice.speaker

            with self.lock:
                try:
                    self.tts.tts_to_file(**kwargs, speed=speed)
                except TypeError:
                    # Older wrappers may not expose speed in tts_to_file even though XTTS
                    # supports it at the model API layer.
                    self.tts.tts_to_file(**kwargs)

            if not output.exists() or output.stat().st_size == 0:
                raise RuntimeError("XTTS did not produce audio")
            return convert_wav(output, response_format)


class Handler(BaseHTTPRequestHandler):
    provider: XttsProvider

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self.send_json(HTTPStatus.OK, self.provider.health())

    def do_POST(self) -> None:
        if self.path != "/v1/audio/speech":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(content_length)
            request = json.loads(body.decode("utf-8"))
            audio, content_type = self.provider.synthesize(request)
            self.send_bytes(HTTPStatus.OK, audio, content_type)
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default=os.environ.get("SIMPLE_AI_XTTS_DEVICE", "auto"))
    parser.add_argument("--coqui-model-name", default=os.environ.get("SIMPLE_AI_XTTS_MODEL", DEFAULT_COQUI_MODEL))
    parser.add_argument("--voices-dir", default=os.environ.get("SIMPLE_AI_XTTS_VOICES_DIR"))
    parser.add_argument("--voice-map", default=os.environ.get("SIMPLE_AI_XTTS_VOICE_MAP"))
    parser.add_argument("--default-language", default=os.environ.get("SIMPLE_AI_XTTS_LANGUAGE", "en"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    Handler.provider = XttsProvider(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"simple-ai-xtts-provider model={args.model} host={args.host} port={args.port} device={Handler.provider.device}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
