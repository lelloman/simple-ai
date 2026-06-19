#!/usr/bin/env python3
"""SimpleAI Chatterbox Multilingual provider.

Chatterbox is MIT licensed and embeds Resemble AI's PerTh watermark in
generated audio. Use reference voices only when you have the rights and consent
to clone them.

The Rust runner starts this process with:
  --model <model-id> --port <port>

The process keeps Chatterbox loaded and exposes:
  GET  /health
  POST /v1/audio/speech

Voice configuration can be provided with --voice-map JSON. Accepted shapes:
  {
    "lello": {"audio_prompt_path": "/voices/lello.wav", "language": "it"},
    "clone": "/voices/clone.wav"
  }

If --voices-dir is set and a requested voice is not in the map, the provider
looks for <voices-dir>/<voice>.wav and uses it as audio_prompt_path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    audio_prompt_path: str | None = None
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


def optional_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def parse_voice_entry(voice_id: str, value: Any) -> VoiceConfig:
    if isinstance(value, str):
        return VoiceConfig(id=voice_id, audio_prompt_path=value)
    if not isinstance(value, dict):
        raise ValueError(f"voice map entry for {voice_id} must be a string or object")

    audio_prompt_path = (
        value.get("audio_prompt_path")
        or value.get("audioPromptPath")
        or value.get("speaker_wav")
        or value.get("speakerWav")
    )
    language = value.get("language")
    return VoiceConfig(
        id=voice_id,
        audio_prompt_path=str(audio_prompt_path) if audio_prompt_path is not None else None,
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


def ensure_path_exists(path: str | None) -> str | None:
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise ValueError(f"audio_prompt_path does not exist: {resolved}")
    return str(resolved)


def convert_wav(wav_path: Path, response_format: str, sample_rate: int) -> tuple[bytes, str]:
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
            str(sample_rate),
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


class ChatterboxProvider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_id = args.model
        self.loaded_at = time.time()
        self.device = choose_device(args.device)
        self.voice_map = load_voice_map(args.voice_map)
        self.voices_dir = Path(args.voices_dir).expanduser() if args.voices_dir else None
        self.lock = threading.Lock()

        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Chatterbox. Install it with: pip install chatterbox-tts"
            ) from exc

        import torch

        self.tts = ChatterboxMultilingualTTS.from_pretrained(
            device=torch.device(self.device),
        )
        self.supported_languages = sorted(ChatterboxMultilingualTTS.get_supported_languages().keys())

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": "chatterbox-multilingual",
            "model": self.model_id,
            "device": self.device,
            "loadedAt": self.loaded_at,
            "sampleRate": self.tts.sr,
            "voices": sorted(self.available_voices()),
            "defaultVoice": self.args.default_voice,
            "defaultLanguage": self.args.default_language,
            "supportedLanguages": self.supported_languages,
        }

    def available_voices(self) -> set[str]:
        voices = {self.args.default_voice, *self.voice_map}
        if self.voices_dir and self.voices_dir.exists():
            voices.update(path.stem for path in self.voices_dir.glob("*.wav"))
        return voices

    def voice_config(self, voice_id: str) -> VoiceConfig:
        if voice_id in self.voice_map:
            return self.voice_map[voice_id]
        if self.voices_dir:
            candidate = self.voices_dir / f"{voice_id}.wav"
            if candidate.exists():
                return VoiceConfig(id=voice_id, audio_prompt_path=str(candidate))
        if voice_id == self.args.default_voice:
            return VoiceConfig(id=voice_id)
        raise ValueError(f"voice {voice_id} is not configured")

    def language_for(self, request: dict[str, Any], voice: VoiceConfig) -> str:
        language = str(
            request.get("language")
            or parse_instruction_language(request.get("instructions"))
            or voice.language
            or self.args.default_language
        ).lower()
        if language not in self.supported_languages:
            raise ValueError(
                f"unsupported language {language}; supported languages: {', '.join(self.supported_languages)}"
            )
        return language

    def synthesize(self, request: dict[str, Any]) -> tuple[bytes, str]:
        if request.get("model") != self.model_id:
            raise ValueError(
                f"request.model {request.get('model')} does not match loaded model {self.model_id}"
            )
        if request.get("stream_format", "audio") == "sse":
            raise ValueError("stream_format=sse is not implemented by the Chatterbox provider")

        text = str(request.get("input") or "").strip()
        if not text:
            raise ValueError("input is required")
        voice_id = parse_voice_id(request.get("voice"))
        voice = self.voice_config(voice_id)
        language = self.language_for(request, voice)
        response_format = normalize_format(request.get("response_format"))
        seed = request.get("seed")

        if seed is not None:
            import numpy as np
            import torch

            seed_int = int(seed)
            random.seed(seed_int)
            np.random.seed(seed_int % (2**32))
            torch.manual_seed(seed_int)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_int)

        generate_kwargs = {
            "text": text,
            "language_id": language,
            "audio_prompt_path": ensure_path_exists(voice.audio_prompt_path),
            "exaggeration": optional_float(request.get("exaggeration"), self.args.exaggeration),
            "cfg_weight": optional_float(request.get("cfg_weight"), self.args.cfg_weight),
            "temperature": optional_float(request.get("temperature"), self.args.temperature),
            "repetition_penalty": optional_float(
                request.get("repetition_penalty"), self.args.repetition_penalty
            ),
            "min_p": optional_float(request.get("min_p"), self.args.min_p),
            "top_p": optional_float(request.get("top_p"), self.args.top_p),
        }

        with tempfile.TemporaryDirectory(prefix="simple_ai_chatterbox_") as tmpdir:
            output = Path(tmpdir) / "speech.wav"
            with self.lock:
                wav = self.tts.generate(**generate_kwargs)

            try:
                import torchaudio as ta
            except Exception as exc:
                raise RuntimeError("torchaudio is required by the Chatterbox provider") from exc

            ta.save(str(output), wav, self.tts.sr)
            if not output.exists() or output.stat().st_size == 0:
                raise RuntimeError("Chatterbox did not produce audio")
            return convert_wav(output, response_format, self.tts.sr)


class Handler(BaseHTTPRequestHandler):
    provider: ChatterboxProvider

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
    parser.add_argument("--device", default=os.environ.get("SIMPLE_AI_CHATTERBOX_DEVICE", "auto"))
    parser.add_argument("--t3-model", default=os.environ.get("SIMPLE_AI_CHATTERBOX_T3_MODEL", "v2"))
    parser.add_argument("--voices-dir", default=os.environ.get("SIMPLE_AI_CHATTERBOX_VOICES_DIR"))
    parser.add_argument("--voice-map", default=os.environ.get("SIMPLE_AI_CHATTERBOX_VOICE_MAP"))
    parser.add_argument("--default-language", default=os.environ.get("SIMPLE_AI_CHATTERBOX_LANGUAGE", "en"))
    parser.add_argument("--default-voice", default=os.environ.get("SIMPLE_AI_CHATTERBOX_DEFAULT_VOICE", "default"))
    parser.add_argument("--exaggeration", type=float, default=float(os.environ.get("SIMPLE_AI_CHATTERBOX_EXAGGERATION", "0.5")))
    parser.add_argument("--cfg-weight", type=float, default=float(os.environ.get("SIMPLE_AI_CHATTERBOX_CFG_WEIGHT", "0.5")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SIMPLE_AI_CHATTERBOX_TEMPERATURE", "0.8")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("SIMPLE_AI_CHATTERBOX_TOP_P", "1.0")))
    parser.add_argument("--min-p", type=float, default=float(os.environ.get("SIMPLE_AI_CHATTERBOX_MIN_P", "0.05")))
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.environ.get("SIMPLE_AI_CHATTERBOX_REPETITION_PENALTY", "2.0")),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    Handler.provider = ChatterboxProvider(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"simple-ai-chatterbox-provider model={args.model} host={args.host} port={args.port} device={Handler.provider.device}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
