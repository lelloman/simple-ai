#!/usr/bin/env python3
"""SimpleAI audio embedding provider for MusicFM and AST.

The Rust runner starts this process with:
  --model <model-id> --port <port>

The process keeps the selected model loaded and exposes:
  GET  /health
  POST /v1/audio/embeddings
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch


AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
MUSICFM_NAMESPACE = "musicfm.mean.v1"
AST_AUDIOSET_NAMESPACE = "ast.audioset.v1"
AST_INSTRUMENTS_NAMESPACE = "ast.instruments.v1"

INSTRUMENT_LABELS = [
    "Singing",
    "Choir",
    "Male singing",
    "Female singing",
    "Plucked string instrument",
    "Guitar",
    "Electric guitar",
    "Bass guitar",
    "Acoustic guitar",
    "Steel guitar, slide guitar",
    "Banjo",
    "Mandolin",
    "Keyboard (musical)",
    "Piano",
    "Electric piano",
    "Organ",
    "Electronic organ",
    "Hammond organ",
    "Synthesizer",
    "Harpsichord",
    "Percussion",
    "Drum kit",
    "Drum machine",
    "Drum",
    "Snare drum",
    "Bass drum",
    "Tabla",
    "Cymbal",
    "Mallet percussion",
    "Orchestra",
    "Brass instrument",
    "Trumpet",
    "Trombone",
    "Bowed string instrument",
    "String section",
    "Violin, fiddle",
    "Cello",
    "Double bass",
    "Flute",
    "Saxophone",
    "Clarinet",
    "Harp",
    "Harmonica",
    "Accordion",
]


def ffmpeg_exe() -> str:
    configured = os.environ.get("SIMPLE_AI_FFMPEG_PATH")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def axis_id(label: str) -> str:
    out = label.lower()
    out = out.replace(", ", "_").replace(" (musical)", "")
    out = "".join(ch if ch.isalnum() else "_" for ch in out)
    while "__" in out:
        out = out.replace("__", "_")
    return "instrument:" + out.strip("_")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def decode_audio(
    input_path: Path,
    sample_rate: int,
    clip_offset_seconds: float | None,
    clip_seconds: float | None,
) -> np.ndarray:
    def run_ffmpeg(offset_seconds: float | None) -> bytes:
        command = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if offset_seconds is not None and offset_seconds > 0:
            command.extend(["-ss", str(offset_seconds)])
        if clip_seconds is not None and clip_seconds > 0:
            command.extend(["-t", str(clip_seconds)])
        command.extend(
            [
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-f",
                "f32le",
                "pipe:1",
            ]
        )
        return subprocess.check_output(command)

    data = run_ffmpeg(clip_offset_seconds)
    audio = np.frombuffer(data, dtype=np.float32)
    if audio.size == 0 and clip_offset_seconds is not None and clip_offset_seconds > 0:
        data = run_ffmpeg(None)
        audio = np.frombuffer(data, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("ffmpeg returned empty audio")
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-12:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def import_musicfm(musicfm_root: Path) -> Any:
    sys.path.insert(0, str(musicfm_root.parent))
    from musicfm.model.musicfm_25hz import MusicFM25Hz

    return MusicFM25Hz


def ensure_min_samples(waveform: torch.Tensor, min_samples: int) -> torch.Tensor:
    if waveform.shape[1] >= min_samples:
        return waveform
    return torch.nn.functional.pad(waveform, (0, min_samples - waveform.shape[1]))


def split_chunks(waveform: torch.Tensor, chunk_secs: float, min_samples: int) -> list[torch.Tensor]:
    chunk_samples = int(24000 * chunk_secs)
    waveform = ensure_min_samples(waveform, min_samples)
    chunks = []
    for start in range(0, waveform.shape[1], chunk_samples):
        chunk = waveform[:, start : start + chunk_samples]
        if chunk.numel():
            chunks.append(ensure_min_samples(chunk, min_samples))
    return chunks


class Provider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_id = args.model
        self.device = choose_device(args.device)
        self.loaded_at = time.time()
        self.musicfm_model = None
        self.ast_extractor = None
        self.ast_model = None
        self.ast_sample_rate = 16000
        self.ast_labels: list[str] = []
        self.ast_label_to_id: dict[str, int] = {}

        if self.model_id == "musicfm-msd":
            self.load_musicfm()
        elif self.model_id == "ast-audioset":
            self.load_ast()
        else:
            raise ValueError(f"unknown audio embedding model: {self.model_id}")

    def load_musicfm(self) -> None:
        musicfm_root = Path(self.args.musicfm_root).expanduser()
        MusicFM25Hz = import_musicfm(musicfm_root)
        self.musicfm_model = MusicFM25Hz(
            is_flash=False,
            stat_path=str(musicfm_root / "data" / "msd_stats.json"),
            model_path=str(musicfm_root / "data" / "pretrained_msd.pt"),
        ).to(self.device)
        self.musicfm_model.eval()

    def load_ast(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.ast_extractor = AutoFeatureExtractor.from_pretrained(self.args.ast_model)
        self.ast_model = AutoModelForAudioClassification.from_pretrained(self.args.ast_model).to(
            self.device
        )
        self.ast_model.eval()
        self.ast_sample_rate = int(getattr(self.ast_extractor, "sampling_rate", 16000) or 16000)
        self.ast_labels = [
            str(self.ast_model.config.id2label[idx])
            for idx in sorted(self.ast_model.config.id2label)
        ]
        self.ast_label_to_id = {label: idx for idx, label in enumerate(self.ast_labels)}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_id,
            "device": str(self.device),
            "loadedAt": self.loaded_at,
        }

    def embed(self, audio_path: Path, options: dict[str, Any]) -> dict[str, Any]:
        namespace = str(options.get("namespace") or "")
        if self.model_id == "musicfm-msd":
            if namespace != MUSICFM_NAMESPACE:
                raise ValueError(f"unsupported MusicFM namespace: {namespace}")
            return self.embed_musicfm(audio_path, options)
        if self.model_id == "ast-audioset":
            if namespace not in {AST_AUDIOSET_NAMESPACE, AST_INSTRUMENTS_NAMESPACE}:
                raise ValueError(f"unsupported AST namespace: {namespace}")
            return self.embed_ast(audio_path, options, namespace)
        raise ValueError(f"unknown model: {self.model_id}")

    def embed_musicfm(self, audio_path: Path, options: dict[str, Any]) -> dict[str, Any]:
        audio = decode_audio(
            audio_path,
            24000,
            options.get("clipOffsetSeconds"),
            options.get("clipSeconds"),
        )
        waveform = torch.from_numpy(audio).unsqueeze(0)
        chunks = split_chunks(
            waveform,
            float(self.args.musicfm_chunk_seconds),
            int(24000 * float(self.args.musicfm_min_seconds)),
        )
        total_count = 0
        sum_vec = None
        with torch.no_grad():
            for batch_start in range(0, len(chunks), int(self.args.musicfm_batch_size)):
                batch_chunks = chunks[batch_start : batch_start + int(self.args.musicfm_batch_size)]
                max_len = max(chunk.shape[1] for chunk in batch_chunks)
                padded = []
                for chunk in batch_chunks:
                    if chunk.shape[1] < max_len:
                        chunk = torch.nn.functional.pad(chunk, (0, max_len - chunk.shape[1]))
                    padded.append(chunk)
                batch = torch.cat(padded, dim=0).to(self.device, non_blocking=True)
                emb = self.musicfm_model.get_latent(
                    batch, layer_ix=int(self.args.musicfm_layer)
                ).float()
                chunk_sum = emb.sum(dim=1).sum(dim=0)
                sum_vec = chunk_sum if sum_vec is None else sum_vec + chunk_sum
                total_count += emb.shape[0] * emb.shape[1]
        if total_count == 0 or sum_vec is None:
            raise ValueError("no MusicFM frames produced")
        mean = (sum_vec / total_count).detach().cpu().numpy().astype(np.float32)
        vector = l2_normalize(mean)
        return {
            "object": "audio_embedding",
            "model": self.model_id,
            "namespace": MUSICFM_NAMESPACE,
            "embedding": vector.tolist(),
            "dim": int(vector.shape[0]),
            "dtype": "float32",
            "metadata": {"numFrames": int(total_count), "normalized": True},
            "modelInfo": {
                "provider": "MusicFM",
                "layer": int(self.args.musicfm_layer),
                "sampleRate": 24000,
            },
        }

    def embed_ast(self, audio_path: Path, options: dict[str, Any], namespace: str) -> dict[str, Any]:
        clip_seconds = options.get("clipSeconds")
        if clip_seconds is None:
            clip_seconds = float(self.args.ast_default_clip_seconds)
        audio = decode_audio(
            audio_path,
            self.ast_sample_rate,
            options.get("clipOffsetSeconds", float(self.args.ast_default_clip_offset_seconds)),
            clip_seconds,
        )
        inputs = self.ast_extractor(
            [audio], sampling_rate=self.ast_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.ast_model(**inputs).logits
            probs = torch.sigmoid(logits[0]).detach().cpu().numpy().astype(np.float32)

        if namespace == AST_AUDIOSET_NAMESPACE:
            vector = probs
            labels = self.ast_labels
        else:
            labels = [label for label in INSTRUMENT_LABELS if label in self.ast_label_to_id]
            vector = np.asarray(
                [probs[self.ast_label_to_id[label]] for label in labels], dtype=np.float32
            )

        top_indices = np.argsort(probs)[-12:][::-1]
        top = [
            {"label": self.ast_labels[int(idx)], "score": float(probs[int(idx)])}
            for idx in top_indices
        ]
        return {
            "object": "audio_embedding",
            "model": self.model_id,
            "namespace": namespace,
            "embedding": vector.tolist(),
            "dim": int(vector.shape[0]),
            "dtype": "float32",
            "metadata": {
                "top": top,
                "labels": labels
                if namespace == AST_AUDIOSET_NAMESPACE
                else [
                    {"id": axis_id(label), "label": label, "column": idx}
                    for idx, label in enumerate(labels)
                ],
            },
            "modelInfo": {
                "provider": "MIT AST AudioSet",
                "model": self.args.ast_model,
                "sampleRate": self.ast_sample_rate,
                "defaultClipOffsetSeconds": float(self.args.ast_default_clip_offset_seconds),
                "defaultClipSeconds": float(self.args.ast_default_clip_seconds),
            },
        }


class Handler(BaseHTTPRequestHandler):
    provider: Provider

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self.send_json(HTTPStatus.OK, self.provider.health())

    def do_POST(self) -> None:
        if self.path != "/v1/audio/embeddings":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            file_name, file_bytes, options = self.read_multipart()
            with tempfile.TemporaryDirectory(prefix="simple_ai_audio_") as tmpdir:
                path = Path(tmpdir) / file_name
                path.write_bytes(file_bytes)
                response = self.provider.embed(path, options)
            self.send_json(HTTPStatus.OK, response)
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
            )

    def read_multipart(self) -> tuple[str, bytes, dict[str, Any]]:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length)
        header = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        )
        msg = BytesParser(policy=default).parsebytes(header + body)

        file_name = "upload.bin"
        file_bytes = None
        options = None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "file":
                file_name = part.get_filename() or file_name
                safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in file_name)
                file_name = safe or "upload.bin"
                file_bytes = part.get_payload(decode=True)
            elif name == "options":
                raw = part.get_payload(decode=True)
                options = json.loads(raw.decode("utf-8"))
        if file_bytes is None:
            raise ValueError("missing multipart file")
        if options is None:
            raise ValueError("missing multipart options")
        if options.get("model") != self.provider.model_id:
            raise ValueError(
                f"options.model {options.get('model')} does not match loaded model {self.provider.model_id}"
            )
        return file_name, file_bytes, options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["musicfm-msd", "ast-audioset"])
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--musicfm-root",
        default="/home/lelloman/rustentia_musicfm/third_party/musicfm",
    )
    parser.add_argument("--musicfm-layer", type=int, default=7)
    parser.add_argument("--musicfm-chunk-seconds", type=float, default=30.0)
    parser.add_argument("--musicfm-min-seconds", type=float, default=1.0)
    parser.add_argument("--musicfm-batch-size", type=int, default=16)
    parser.add_argument("--ast-model", default=AST_MODEL)
    parser.add_argument("--ast-default-clip-offset-seconds", type=float, default=30.0)
    parser.add_argument("--ast-default-clip-seconds", type=float, default=10.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    Handler.provider = Provider(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"simple-ai-audio-provider model={args.model} host={args.host} port={args.port} device={Handler.provider.device}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
