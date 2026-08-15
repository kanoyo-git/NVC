"""NVC inference worker for the NVC Realtime VST2/VST3 plugin."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import struct
import sys
import time
import traceback
from pathlib import Path

MAGIC = 0x50564E43
PROTOCOL_VERSION = 1
HEADER_BYTES = 4096
MAX_FRAMES = 131072
MAP_BYTES = HEADER_BYTES + MAX_FRAMES * 4 * 2
INPUT_OFFSET = HEADER_BYTES
OUTPUT_OFFSET = HEADER_BYTES + MAX_FRAMES * 4
STATUS_TEXT_OFFSET = 128
STATUS_TEXT_BYTES = 512

STATUS_STARTING = 1
STATUS_LOADING = 2
STATUS_READY = 3
STATUS_ERROR = -1
STATUS_STOP = -2

SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258


def write_value(shared: mmap.mmap, offset: int, fmt: str, value) -> None:
    struct.pack_into("<" + fmt, shared, offset, value)


def read_value(shared: mmap.mmap, offset: int, fmt: str):
    return struct.unpack_from("<" + fmt, shared, offset)[0]


def write_status(shared: mmap.mmap, state: int, message: str) -> None:
    write_value(shared, 8, "i", state)
    encoded = message.encode("utf-8", errors="replace")[: STATUS_TEXT_BYTES - 1]
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + STATUS_TEXT_BYTES] = b"\0" * STATUS_TEXT_BYTES
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + len(encoded)] = encoded
    shared.flush()


class WinEvent:
    def __init__(self, name: str):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        kernel32.OpenEventW.restype = ctypes.c_void_p
        self._wait = kernel32.WaitForSingleObject
        self._wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._wait.restype = ctypes.c_uint32
        self._set = kernel32.SetEvent
        self._set.argtypes = [ctypes.c_void_p]
        self._set.restype = ctypes.c_int
        self._close = kernel32.CloseHandle
        self._close.argtypes = [ctypes.c_void_p]
        self._close.restype = ctypes.c_int
        self.handle = kernel32.OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, False, name)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), f"OpenEventW failed: {name}")

    def wait(self, timeout_ms: int) -> int:
        return int(self._wait(self.handle, timeout_ms))

    def set(self) -> None:
        if not self._set(self.handle):
            raise OSError(ctypes.get_last_error(), "SetEvent failed")

    def close(self) -> None:
        if self.handle:
            self._close(self.handle)
            self.handle = None


class NVCStreamEngine:
    def __init__(self, cfg: dict):
        self.root = Path(cfg["nvc_root"]).resolve()
        os.chdir(self.root)
        sys.path.insert(0, str(self.root))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "4")

        import librosa
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torchaudio.transforms as tat

        from configs.config import Config
        from infer import rtnvc
        from tools.cuda_graph import run_cuda_graph

        self.librosa = librosa
        self.np = np
        self.torch = torch
        self.F = F
        self.tat = tat
        self.run_cuda_graph = run_cuda_graph

        self.sample_rate = int(cfg["sample_rate"])
        self.block_ms = float(cfg["block_ms"])
        self.crossfade_ms = float(cfg["crossfade_ms"])
        self.extra_ms = float(cfg["extra_ms"])
        self.zc = max(1, self.sample_rate // 100)
        self.block_frame = int(round(self.block_ms / 1000 * self.sample_rate / self.zc) * self.zc)
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(round(self.crossfade_ms / 1000 * self.sample_rate / self.zc) * self.zc)
        # Match the source realtime GUI: SOLA overlap is capped at 40 ms and
        # remains independent of the audio callback block length.
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.effective_crossfade_ms = 1000.0 * self.sola_buffer_frame / self.sample_rate
        self.sola_search_frame = self.zc
        self.extra_frame = int(round(self.extra_ms / 1000 * self.sample_rate / self.zc) * self.zc)

        self.config = Config()
        self.nvc = rtnvc.NVC(
            0.0,
            0.0,
            str(Path(cfg["model_path"]).resolve()),
            str(Path(cfg["index_path"]).resolve()) if cfg.get("index_path") else "",
            0.0,
            self.config,
            None,
        )

        total_frames = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(total_frames, device=self.config.device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(160 * total_frames // self.zc, device=self.config.device, dtype=torch.float32)
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = torch.sin(
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device)
        ) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(orig_freq=self.sample_rate, new_freq=16000, dtype=torch.float32).to(self.config.device)
        self.resampler2 = None
        if self.nvc.tgt_sr != self.sample_rate:
            self.resampler2 = tat.Resample(orig_freq=self.nvc.tgt_sr, new_freq=self.sample_rate, dtype=torch.float32).to(self.config.device)
        self.last_pitch = None
        self.last_formant = None
        self.last_index_rate = None

    def prewarm(self) -> None:
        phase = self.torch.arange(self.block_frame, device=self.config.device, dtype=self.torch.float32)
        probe = 0.05 * self.torch.sin(2 * self.np.pi * 220.0 * phase / self.sample_rate)
        self.process(probe.cpu().numpy(), 12.0, 0.0, 0.0, 0.5, -60.0, 0)
        if self.torch.device(self.config.device).type == "cuda":
            self.torch.cuda.synchronize(self.config.device)

        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.rms_buffer.fill(0.0)
        self.sola_buffer.zero_()
        if hasattr(self.nvc, "cache_pitch"):
            self.nvc.cache_pitch.zero_()
        if hasattr(self.nvc, "cache_pitchf"):
            self.nvc.cache_pitchf.zero_()

    def process(self, audio, pitch: float, formant: float, index_rate: float,
                rms_mix: float, threshold: float, f0_method: int):
        np = self.np
        torch = self.torch
        F = self.F
        if len(audio) != self.block_frame:
            raise ValueError(f"block mismatch: worker={self.block_frame}, request={len(audio)}")

        if self.last_pitch != pitch:
            self.nvc.change_key(pitch)
            self.last_pitch = pitch
        if self.last_formant != formant:
            self.nvc.change_formant(formant)
            self.last_formant = formant
        if self.last_index_rate != index_rate:
            self.nvc.change_index_rate(index_rate)
            self.last_index_rate = index_rate

        indata = np.asarray(audio, dtype=np.float32).copy()
        if threshold > -60.0:
            gated = np.append(self.rms_buffer, indata)
            rms = self.librosa.feature.rms(y=gated, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
            self.rms_buffer[:] = gated[-4 * self.zc :]
            gated = gated[2 * self.zc - self.zc // 2 :]
            below = self.librosa.amplitude_to_db(rms, ref=1.0)[0] < threshold
            for i, mute in enumerate(below):
                if mute:
                    gated[i * self.zc : (i + 1) * self.zc] = 0
            indata = gated[self.zc // 2 :]

        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
        resampled = self.run_cuda_graph(
            self.resampler,
            "vst-input-resample",
            lambda value: self.resampler(value),
            resample_input,
        )[160:]
        self.input_wav_res[-self.block_frame_16k:] = resampled[-self.block_frame_16k:]

        method = ("rmvpe", "fcpe", "pm")[max(0, min(2, int(f0_method)))]
        infer_wav = self.nvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            method,
        )
        if self.resampler2 is not None:
            infer_wav = self.run_cuda_graph(
                self.resampler2,
                "vst-output-resample",
                lambda value: self.resampler2(value),
                infer_wav,
            )

        if rms_mix < 1.0:
            input_tail = self.input_wav[self.extra_frame :]
            rms1 = self.librosa.feature.rms(
                y=input_tail[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = self.librosa.feature.rms(
                y=infer_wav.cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.maximum(rms2, torch.full_like(rms2, 1e-3))
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - rms_mix)

        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        return infer_wav[: self.block_frame].float().cpu().numpy()


def run(args: argparse.Namespace) -> int:
    shared = None
    request_event = None
    response_event = None
    try:
        shared = mmap.mmap(-1, MAP_BYTES, tagname=args.map, access=mmap.ACCESS_WRITE)
        request_event = WinEvent(args.request)
        response_event = WinEvent(args.response)
        if read_value(shared, 0, "I") != MAGIC or read_value(shared, 4, "I") != PROTOCOL_VERSION:
            raise RuntimeError("NVC VST protocol mismatch")
        write_status(shared, STATUS_LOADING, "Loading Python and CUDA")
        with open(args.config, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        # NVC's Config parses process-wide CLI flags intended for its GUI.
        sys.argv = [sys.argv[0]]
        engine = NVCStreamEngine(cfg)
        write_status(shared, STATUS_LOADING, "Prewarming CUDA and F0")
        engine.prewarm()
        write_status(shared, STATUS_READY, f"Ready (actual CF {engine.effective_crossfade_ms:.0f} ms)")

        import numpy as np

        input_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=INPUT_OFFSET)
        output_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=OUTPUT_OFFSET)
        last_sequence = 0
        while True:
            result = request_event.wait(1000)
            if result == WAIT_TIMEOUT:
                continue
            if result != WAIT_OBJECT_0:
                raise OSError("WaitForSingleObject failed")
            if read_value(shared, 8, "i") == STATUS_STOP:
                break
            sequence = read_value(shared, 12, "I")
            if sequence == last_sequence:
                continue
            frames = read_value(shared, 20, "I")
            if frames <= 0 or frames > MAX_FRAMES:
                raise ValueError(f"invalid frame count: {frames}")
            started = time.perf_counter()
            processed = engine.process(
                input_view[:frames],
                read_value(shared, 32, "f"),
                read_value(shared, 36, "f"),
                read_value(shared, 40, "f"),
                read_value(shared, 44, "f"),
                read_value(shared, 48, "f"),
                read_value(shared, 64, "I"),
            )
            output_view[:frames] = processed
            write_value(shared, 56, "f", (time.perf_counter() - started) * 1000.0)
            write_value(shared, 16, "I", sequence)
            write_value(shared, 8, "i", STATUS_READY)
            last_sequence = sequence
            response_event.set()
        return 0
    except BaseException:
        message = traceback.format_exc()
        try:
            Path(args.config + ".log").write_text(message, encoding="utf-8")
        except Exception:
            pass
        try:
            if shared is not None:
                write_status(shared, STATUS_ERROR, message[-500:])
            if response_event is not None:
                response_event.set()
        except Exception:
            pass
        return 1
    finally:
        if request_event is not None:
            request_event.close()
        if response_event is not None:
            response_event.close()
        if shared is not None:
            shared.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
