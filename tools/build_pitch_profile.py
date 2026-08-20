"""Build an RVC model pitch-profile sidecar from its training vocal."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer.audio import load_audio
from infer.rmvpe import RMVPE
from infer.vc.pitch_tuning import hz_to_midi, stabilize_f0


QUANTILES = (
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
)
AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
}


class StreamingPitchStats:
    """Bounded-memory pitch statistics for datasets of arbitrary duration."""

    def __init__(self, bin_count=4096):
        minimum = float(hz_to_midi(50.0))
        maximum = float(hz_to_midi(1100.0))
        self.edges = np.linspace(minimum, maximum, int(bin_count) + 1)
        self.histogram = np.zeros(int(bin_count), dtype=np.int64)
        self.total_frames = 0
        self.voiced_frames = 0
        self.midi_sum = 0.0
        self.midi_square_sum = 0.0

    def update(self, f0):
        f0 = np.asarray(f0, dtype=np.float64)
        self.total_frames += int(f0.size)
        voiced = f0[np.isfinite(f0) & (f0 >= 50.0) & (f0 <= 1100.0)]
        if not voiced.size:
            return
        midi = hz_to_midi(voiced)
        self.histogram += np.histogram(midi, bins=self.edges)[0]
        self.voiced_frames += int(midi.size)
        self.midi_sum += float(np.sum(midi))
        self.midi_square_sum += float(np.sum(np.square(midi)))

    def quantiles(self, probabilities):
        if self.voiced_frames == 0:
            raise RuntimeError("Dataset contains no usable voiced F0 frames")
        cumulative = np.cumsum(self.histogram)
        values = []
        for probability in probabilities:
            target = float(probability) * max(self.voiced_frames - 1, 0)
            index = min(
                int(np.searchsorted(cumulative, target, side="right")),
                len(self.histogram) - 1,
            )
            previous = int(cumulative[index - 1]) if index else 0
            count = max(int(self.histogram[index]), 1)
            fraction = np.clip((target - previous) / count, 0.0, 1.0)
            values.append(
                self.edges[index]
                + fraction * (self.edges[index + 1] - self.edges[index])
            )
        return np.asarray(values)

    @property
    def mean(self):
        return self.midi_sum / max(self.voiced_frames, 1)

    @property
    def standard_deviation(self):
        variance = self.midi_square_sum / max(self.voiced_frames, 1) - self.mean**2
        return math.sqrt(max(variance, 0.0))


def collect_audio_files(source):
    source = Path(source).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError("Unsupported dataset audio file: %s" % source)
        return source, [source]
    if not source.is_dir():
        raise FileNotFoundError("Dataset path not found: %s" % source)
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise RuntimeError("Dataset directory contains no supported audio files")
    return source, files


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract a model-aware pitch profile from training audio."
    )
    parser.add_argument("dataset", help="Training vocal file or dataset directory")
    parser.add_argument("--name", help="Profile/voice name")
    parser.add_argument("--output", help="JSON sidecar; stdout when omitted")
    parser.add_argument(
        "--model",
        help="Model .pth; writes MODEL.pitch.json beside it unless --output is set",
    )
    parser.add_argument("--device", help="Torch device; auto-selected when omitted")
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.03)
    return parser


def extract_chunked(model, audio, chunk_seconds, threshold, sample_rate=16000):
    chunk = max(int(sample_rate * chunk_seconds), sample_rate)
    overlap = sample_rate
    parts = []
    for start in range(0, len(audio), chunk - overlap):
        end = min(start + chunk, len(audio))
        f0 = model.infer_from_audio(audio[start:end], thred=threshold)
        trim = 0 if start == 0 else overlap // 160 // 2
        trim_end = None if end == len(audio) else -(overlap // 160 - trim)
        parts.append(f0[trim:trim_end])
        if end == len(audio):
            break
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def build_pitch_profile(
    dataset,
    name=None,
    device=None,
    chunk_seconds=20.0,
    threshold=0.03,
    progress=None,
):
    """Build a reusable pitch profile from one file or a dataset directory.

    ``progress`` receives ``(index, total, audio_path)`` and lets both the CLI
    and Studio present progress without duplicating the extraction code.
    """

    chunk_seconds = float(chunk_seconds)
    threshold = float(threshold)
    if chunk_seconds <= 1.0:
        raise ValueError("chunk_seconds must be greater than one")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    dataset_path, audio_files = collect_audio_files(dataset)
    profile_name = str(name or dataset_path.stem)
    is_half = bool(device and str(device).startswith("cuda"))
    model = RMVPE(
        str(PROJECT_ROOT / "assets" / "rmvpe" / "rmvpe.pt"),
        is_half=is_half,
        device=device,
    )
    statistics = StreamingPitchStats()
    for index, audio_path in enumerate(audio_files, 1):
        if progress is not None:
            progress(index, len(audio_files), audio_path)
        audio = load_audio(str(audio_path), 16000)
        statistics.update(
            stabilize_f0(
                extract_chunked(model, audio, chunk_seconds, threshold)
            )
        )
    if statistics.voiced_frames < 100:
        raise RuntimeError("Training audio contains too few voiced F0 frames")
    values = statistics.quantiles(QUANTILES)
    profile = {
        "schema_version": 1,
        "name": profile_name,
        "source": dataset_path.name,
        "audio_files": len(audio_files),
        "extractor": {
            "name": "rmvpe",
            "sample_rate": 16000,
            "hop_length": 160,
            "threshold": threshold,
        },
        "total_frames": statistics.total_frames,
        "voiced_frames": statistics.voiced_frames,
        "voiced_ratio": round(
            statistics.voiced_frames / max(statistics.total_frames, 1), 6
        ),
        "mean_midi": round(statistics.mean, 4),
        "std_midi": round(statistics.standard_deviation, 4),
        "midi_quantiles": {
            f"{quantile:g}": round(float(value), 4)
            for quantile, value in zip(QUANTILES, values)
        },
    }
    return profile


def save_pitch_profile(profile, output):
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_path = Path(args.model).expanduser().resolve() if args.model else None
    if model_path is not None and (
        not model_path.is_file() or model_path.suffix.lower() != ".pth"
    ):
        raise ValueError("--model must point to an existing .pth file")
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if output_path is None and model_path is not None:
        output_path = model_path.with_suffix(".pitch.json")
    dataset_path = Path(args.dataset).expanduser().resolve()
    profile_name = args.name or (model_path.stem if model_path else dataset_path.stem)

    def print_progress(index, total, audio_path):
        print(
            "[%d/%d] %s" % (index, total, audio_path.name),
            file=sys.stderr,
            flush=True,
        )

    profile = build_pitch_profile(
        args.dataset,
        name=profile_name,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        threshold=args.threshold,
        progress=print_progress,
    )
    text = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
    if output_path is not None:
        save_pitch_profile(profile, output_path)
        print(str(output_path))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
