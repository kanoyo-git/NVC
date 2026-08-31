"""NMVPE pitch extraction for NVC.

NMVPE keeps RMVPE's frontend and checkpoint layout, adds the trained timbre
filter, and decodes pitch with an explicit voiced/unvoiced temporal path.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from infer.rmvpe import ConvBlockRes, Decoder, E2E, Encoder, Intermediate, MelSpectrogram


SAMPLE_RATE = 16_000
N_CLASS = 360
N_MELS = 128
CENTS_PER_BIN = 20.0
CENTS_OFFSET = 1997.3794084376191


class TimbreFilter(nn.Module):
    """Filter encoder skip features before they reach the pitch decoder."""

    def __init__(self, latent_channels):
        super().__init__()
        self.layers = nn.ModuleList(
            ConvBlockRes(channels, channels) for channels, _ in latent_channels
        )

    def forward(self, tensors):
        return [layer(tensor) for layer, tensor in zip(self.layers, tensors)]


class FilteredDeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            N_MELS,
            en_de_layers,
            kernel_size,
            n_blocks,
            en_out_channels,
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, inputs):
        encoded, skip_features = self.encoder(inputs)
        encoded = self.intermediate(encoded)
        return self.decoder(encoded, self.tf(skip_features))


class FilteredE2E(E2E):
    """RMVPE-compatible pitch head backed by NMVPE's filtered U-Net."""

    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__(
            n_blocks,
            n_gru,
            kernel_size,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.unet = FilteredDeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def _strip_data_parallel_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _local_average_f0(salience, centers, voiced):
    mapping = CENTS_PER_BIN * np.arange(N_CLASS, dtype=np.float64) + CENTS_OFFSET
    output = np.zeros(salience.shape[0], dtype=np.float32)
    for frame, center in enumerate(centers):
        if not voiced[frame]:
            continue
        start = max(0, int(center) - 4)
        end = min(N_CLASS, int(center) + 5)
        weights = np.clip(salience[frame, start:end], 0.0, None)
        weight_sum = float(weights.sum())
        cents = (
            float(np.sum(weights * mapping[start:end]) / weight_sum)
            if weight_sum > 1e-12
            else float(mapping[int(center)])
        )
        output[frame] = 10.0 * 2.0 ** (cents / 1200.0)
    return output


def _pitch_candidates(salience, count, context=2):
    frames, bins = salience.shape
    count = max(1, min(int(count), bins))
    left = np.pad(salience[:, :-1], ((0, 0), (1, 0)), constant_values=-np.inf)
    right = np.pad(salience[:, 1:], ((0, 0), (0, 1)), constant_values=-np.inf)
    peak_scores = np.where(
        (salience >= left) & (salience >= right), salience, -np.inf
    )
    top = np.argpartition(peak_scores, bins - count, axis=1)[:, -count:]
    top_scores = np.take_along_axis(peak_scores, top, axis=1)
    top = np.take_along_axis(
        top, np.argsort(top_scores, axis=1)[:, ::-1], axis=1
    )
    strongest = np.argmax(salience, axis=1)

    candidates = np.empty((frames, count), dtype=np.int16)
    for frame in range(frames):
        selected = []
        first = max(0, frame - context)
        last = min(frames, frame + context + 1)
        for neighbour in range(first, last):
            value = int(strongest[neighbour])
            if value not in selected:
                selected.append(value)
        for value in top[frame]:
            value = int(value)
            if value not in selected:
                selected.append(value)
            if len(selected) == count:
                break
        if len(selected) < count:
            for value in np.argsort(salience[frame])[::-1]:
                value = int(value)
                if value not in selected:
                    selected.append(value)
                if len(selected) == count:
                    break
        candidates[frame] = selected[:count]
    return candidates


def _track_pitch_path(
    salience,
    threshold=0.03,
    candidate_count=12,
    emission_temperature=0.02,
    jump_cost=0.06,
    voiced_switch_cost=2.0,
):
    """Track local pitch peaks plus an explicit unvoiced state."""

    salience = np.asarray(salience, dtype=np.float64)
    if salience.ndim != 2 or salience.shape[1] != N_CLASS:
        raise ValueError(f"salience must have shape [frames, {N_CLASS}]")
    if salience.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if emission_temperature <= 0:
        raise ValueError("emission_temperature must be positive")

    safe = np.clip(
        np.nan_to_num(salience, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )
    candidates = _pitch_candidates(safe, candidate_count)
    frames, candidate_count = candidates.shape
    scores = np.take_along_axis(safe, candidates, axis=1)
    pitch_emission = (scores - threshold) / emission_temperature
    unvoiced_emission = (threshold - safe.max(axis=1)) / emission_temperature

    back_pitch = np.empty((frames, candidate_count), dtype=np.int16)
    back_unvoiced = np.empty(frames, dtype=np.int16)
    previous_pitch = pitch_emission[0].copy()
    previous_unvoiced = unvoiced_emission[0]
    back_pitch[0].fill(candidate_count)
    back_unvoiced[0] = candidate_count

    for frame in range(1, frames):
        distance = np.abs(
            candidates[frame - 1].astype(np.float64)[:, None]
            - candidates[frame].astype(np.float64)[None, :]
        )
        transitions = previous_pitch[:, None] - jump_cost * distance
        best_positions = np.argmax(transitions, axis=0)
        best_scores = transitions[best_positions, np.arange(candidate_count)]
        from_unvoiced = previous_unvoiced - voiced_switch_cost
        use_unvoiced = from_unvoiced > best_scores
        current_pitch = pitch_emission[frame] + np.where(
            use_unvoiced, from_unvoiced, best_scores
        )
        back_pitch[frame] = np.where(
            use_unvoiced, candidate_count, best_positions
        )

        best_voiced_position = int(np.argmax(previous_pitch))
        from_voiced = previous_pitch[best_voiced_position] - voiced_switch_cost
        if from_voiced > previous_unvoiced:
            current_unvoiced = unvoiced_emission[frame] + from_voiced
            back_unvoiced[frame] = best_voiced_position
        else:
            current_unvoiced = unvoiced_emission[frame] + previous_unvoiced
            back_unvoiced[frame] = candidate_count

        previous_pitch = current_pitch
        previous_unvoiced = current_unvoiced

    state = (
        int(np.argmax(previous_pitch))
        if previous_pitch.max() > previous_unvoiced
        else candidate_count
    )
    centers = np.zeros(frames, dtype=np.int64)
    voiced = np.zeros(frames, dtype=bool)
    for frame in range(frames - 1, -1, -1):
        if state == candidate_count:
            state = int(back_unvoiced[frame])
        else:
            voiced[frame] = True
            centers[frame] = int(candidates[frame, state])
            state = int(back_pitch[frame, state])
    return centers, voiced


def to_viterbi_f0(hidden, thred=0.03, **tracker_kwargs):
    if torch.is_tensor(hidden):
        salience = hidden.detach().float().cpu().numpy()
    else:
        salience = np.asarray(hidden, dtype=np.float32)
    if salience.ndim == 2:
        salience = salience[None, ...]
    if salience.ndim != 3 or salience.shape[-1] != N_CLASS:
        raise ValueError(f"hidden must have shape [batch, frames, {N_CLASS}]")

    decoded = []
    for item in salience:
        centers, voiced = _track_pitch_path(
            item, threshold=thred, **tracker_kwargs
        )
        decoded.append(_local_average_f0(item, centers, voiced))
    result = np.stack(decoded, axis=0)
    return result[0] if result.shape[0] == 1 else result


def to_local_average_f0(hidden, thred=0.03):
    if not torch.is_tensor(hidden):
        hidden = torch.as_tensor(hidden)
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    if hidden.ndim != 3 or hidden.shape[-1] != N_CLASS:
        raise ValueError(f"hidden must have shape [batch, frames, {N_CLASS}]")

    safe = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=0.0)
    indices = torch.arange(N_CLASS, device=hidden.device)[None, None, :]
    centers = torch.argmax(safe, dim=2, keepdim=True)
    weights = safe * (
        (indices >= torch.clamp(centers - 4, min=0))
        & (indices < torch.clamp(centers + 5, max=N_CLASS))
    )
    cents = torch.sum(
        weights * (indices * CENTS_PER_BIN + CENTS_OFFSET), dim=2
    ) / torch.sum(weights, dim=2).clamp_min(1e-12)
    f0 = 10.0 * 2.0 ** (cents / 1200.0)
    f0 = torch.where(safe.amax(dim=2) >= thred, f0, torch.zeros_like(f0))
    result = f0.detach().cpu().numpy().astype(np.float32)
    return result[0] if result.shape[0] == 1 else result


class NMVPE:
    """Load and run NMVPE without replacing NVC's original RMVPE method."""

    def __init__(
        self,
        model_path,
        is_half=False,
        device=None,
        hop_length=160,
        model_variant=None,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"NMVPE checkpoint not found: {model_path}. "
                "Place NMVPE_stable_v1.pt there as nmvpe.pt or set nmvpe_root."
            )
        if not isinstance(hop_length, int) or hop_length <= 0:
            raise ValueError("hop_length must be a positive integer")
        if isinstance(is_half, str):
            is_half = is_half.lower() == "true"
        if device is None:
            from configs.config import infer_device, infer_dtype

            device = infer_device
            is_half = infer_dtype == torch.float16
        elif str(device).startswith("cuda"):
            from configs.config import get_device_dtype_sm

            parsed_device = torch.device(device)
            index = parsed_device.index
            if index is None:
                index = torch.cuda.current_device()
            device, dtype, _, _ = get_device_dtype_sm(index)
            is_half = dtype == torch.float16
        else:
            is_half = False

        checkpoint = _load_checkpoint(model_path)
        state_dict = checkpoint.get("model", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("NMVPE checkpoint must contain a model state dict")
        state_dict = _strip_data_parallel_prefix(state_dict)
        detected_variant = (
            "filtered" if any("unet.tf." in key for key in state_dict) else "legacy"
        )
        self.model_variant = model_variant or checkpoint.get(
            "model_variant", detected_variant
        )
        if self.model_variant not in {"legacy", "filtered"}:
            raise ValueError("model_variant must be 'legacy' or 'filtered'")

        model_class = FilteredE2E if self.model_variant == "filtered" else E2E
        self.model = model_class(4, 1, (2, 2))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        decoder_config = checkpoint.get("decoder_config", {})
        if not isinstance(decoder_config, dict):
            raise ValueError("decoder_config must be a dictionary")
        allowed = {
            "thred",
            "candidate_count",
            "emission_temperature",
            "jump_cost",
            "voiced_switch_cost",
        }
        unexpected = sorted(set(decoder_config) - allowed)
        if unexpected:
            raise ValueError(
                "unsupported decoder_config keys: %s" % ", ".join(unexpected)
            )
        self.decoder_config = dict(decoder_config)

        self.device = device
        self.is_half = bool(is_half)
        self.hop_length = hop_length
        self.model = (
            self.model.half() if self.is_half else self.model.float()
        ).to(self.device)
        self.mel_extractor = MelSpectrogram(
            self.is_half,
            N_MELS,
            SAMPLE_RATE,
            1024,
            hop_length,
            None,
            30,
            8000,
        ).to(self.device)
        self.resample_kernel = {}

    def _forward_mel(self, mel):
        n_frames = mel.shape[-1]
        if n_frames == 0:
            return mel.new_empty((mel.shape[0], 0, N_CLASS))
        padded_frames = 32 * ((n_frames - 1) // 32 + 1)
        padding = padded_frames - n_frames
        if padding:
            mode = "reflect" if n_frames > 1 and padding < n_frames else "replicate"
            mel = F.pad(mel, (0, padding), mode=mode)
        mel = mel.half() if self.is_half else mel.float()
        return self.model(mel)[:, :n_frames]

    def mel2hidden(self, mel, chunk_frames=4096, context_frames=128):
        if mel.ndim != 3 or mel.shape[1] != N_MELS:
            raise ValueError(f"mel must have shape [batch, {N_MELS}, frames]")
        if chunk_frames is None or mel.shape[-1] <= chunk_frames:
            with torch.inference_mode():
                return self._forward_mel(mel)
        if chunk_frames <= 0 or context_frames < 0:
            raise ValueError(
                "chunk_frames must be positive and context_frames non-negative"
            )

        outputs = []
        frames = mel.shape[-1]
        with torch.inference_mode():
            for core_start in range(0, frames, chunk_frames):
                core_end = min(frames, core_start + chunk_frames)
                chunk_start = max(0, core_start - context_frames)
                chunk_end = min(frames, core_end + context_frames)
                hidden = self._forward_mel(mel[..., chunk_start:chunk_end])
                outputs.append(
                    hidden[:, core_start - chunk_start : core_end - chunk_start]
                )
        return torch.cat(outputs, dim=1)

    def decode(self, hidden, thred=None, decoder="stable", **decoder_kwargs):
        if decoder in {"stable", "viterbi"}:
            config = dict(self.decoder_config)
            configured_threshold = config.pop("thred", 0.03)
            threshold = configured_threshold if thred is None else thred
            config.update(decoder_kwargs)
            if not 0.0 < float(threshold) < 1.0:
                raise ValueError("thred must be between 0 and 1")
            return to_viterbi_f0(hidden, thred=threshold, **config)
        if decoder == "local":
            if decoder_kwargs:
                raise TypeError(
                    "local decoder does not accept: %s"
                    % ", ".join(sorted(decoder_kwargs))
                )
            threshold = 0.03 if thred is None else thred
            if not 0.0 < float(threshold) < 1.0:
                raise ValueError("thred must be between 0 and 1")
            return to_local_average_f0(hidden, thred=threshold)
        raise ValueError("decoder must be 'stable' or 'local'")

    @staticmethod
    def _prepare_audio(audio):
        if torch.is_tensor(audio):
            waveform = audio.detach().float()
        else:
            waveform = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        waveform = waveform.squeeze()
        if waveform.ndim == 0:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim == 2:
            if waveform.shape[0] <= 8:
                waveform = waveform.mean(dim=0)
            elif waveform.shape[1] <= 8:
                waveform = waveform.mean(dim=1)
            else:
                raise ValueError("2-D audio must have a recognizable channel axis")
        if waveform.ndim != 1:
            raise ValueError("audio must be mono or multi-channel")
        if waveform.numel() == 0:
            raise ValueError("audio is empty")
        if not torch.isfinite(waveform).all():
            raise ValueError("audio contains NaN or infinity")
        return waveform

    @staticmethod
    def _normalize_audio(waveform, target_dbfs=-20.0, max_gain=100.0):
        rms = waveform.square().mean().sqrt()
        peak = waveform.abs().max()
        if rms <= 1e-7 or peak <= 1e-7:
            return waveform
        target_rms = 10.0 ** (target_dbfs / 20.0)
        gain = torch.minimum(target_rms / rms, 0.99 / peak)
        return waveform * torch.clamp(gain, max=max_gain)

    def infer_from_audio(
        self,
        audio,
        sample_rate=SAMPLE_RATE,
        thred=None,
        decoder="stable",
        normalize_audio=True,
        chunk_frames=4096,
        context_frames=128,
        return_confidence=False,
        **decoder_kwargs,
    ):
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        waveform = self._prepare_audio(audio)
        if normalize_audio:
            waveform = self._normalize_audio(waveform)
        waveform = waveform.unsqueeze(0).to(self.device)
        if sample_rate != SAMPLE_RATE:
            key = (sample_rate, str(self.device))
            if key not in self.resample_kernel:
                self.resample_kernel[key] = Resample(
                    sample_rate, SAMPLE_RATE, lowpass_filter_width=128
                ).to(self.device)
            waveform = self.resample_kernel[key](waveform)

        with torch.inference_mode():
            mel = self.mel_extractor(waveform, center=True)
            hidden = self.mel2hidden(
                mel,
                chunk_frames=chunk_frames,
                context_frames=context_frames,
            )
        f0 = self.decode(hidden, thred=thred, decoder=decoder, **decoder_kwargs)
        if return_confidence:
            confidence = hidden.amax(dim=-1).squeeze(0).float().cpu().numpy()
            return f0, confidence.astype(np.float32)
        return f0
