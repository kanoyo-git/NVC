"""Adaptive full-band source-detail preservation for NVC inference."""

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d


def _match_length(audio, length):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == length:
        return audio
    if audio.size < 2 or length < 2:
        return np.zeros(length, dtype=np.float32)
    source_axis = np.linspace(0.0, 1.0, audio.size, dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return np.interp(target_axis, source_axis, audio).astype(np.float32)


def _frame_voicing(voiced, frame_count):
    if voiced is None:
        return None
    voiced = np.asarray(voiced, dtype=np.float32).reshape(-1)
    if voiced.size == 0:
        return None
    if voiced.size == 1:
        return np.full(frame_count, np.clip(voiced[0], 0.0, 1.0), np.float32)
    positions = np.linspace(0.0, voiced.size - 1, frame_count)
    return np.interp(positions, np.arange(voiced.size), voiced).astype(np.float32)


def adaptive_detail_morph(converted, source, sample_rate, amount, voiced=None):
    """Blend recolored aperiodic source detail into a converted waveform.

    ``amount`` is the maximum residual contribution. Harmonic/voiced regions
    are strongly suppressed so pitch transposition does not leak the source
    note into the converted signal.
    """
    converted = np.asarray(converted, dtype=np.float32).reshape(-1)
    amount = float(np.clip(amount, 0.0, 0.25))
    if amount <= 0.0 or converted.size < 64:
        return converted.copy()

    source = _match_length(source, converted.size)
    n_fft = 1024 if int(sample_rate) >= 32000 else 512
    if converted.size < n_fft:
        n_fft = 2 ** int(np.floor(np.log2(max(64, converted.size))))
    hop_length = max(16, n_fft // 4)

    generated_spec = librosa.stft(
        converted, n_fft=n_fft, hop_length=hop_length, win_length=n_fft
    )
    source_spec = librosa.stft(
        source, n_fft=n_fft, hop_length=hop_length, win_length=n_fft
    )
    generated_mag = np.abs(generated_spec).astype(np.float32)
    source_mag = np.abs(source_spec).astype(np.float32)
    eps = np.float32(1e-5)

    # Whitening keeps the source timing/phase microstructure. Recoloring with
    # the generated envelope avoids copying the source speaker's formants.
    envelope_bins = max(5, int(round(n_fft * 500.0 / sample_rate)) | 1)
    source_envelope = uniform_filter1d(
        source_mag, size=envelope_bins, axis=0, mode="nearest"
    )
    generated_envelope = uniform_filter1d(
        generated_mag, size=envelope_bins, axis=0, mode="nearest"
    )
    color_envelope = np.maximum(generated_envelope, source_envelope * 0.75)
    recolored_source = source_spec / (source_envelope + eps) * color_envelope

    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    detail_band = np.clip((frequencies - 700.0) / 2300.0, 0.0, 1.0)
    analysis_band = (frequencies >= 900.0) & (
        frequencies <= min(14000.0, sample_rate * 0.48)
    )
    band_mag = source_mag[analysis_band]
    if band_mag.size:
        flatness = np.exp(np.mean(np.log(band_mag + eps), axis=0)) / (
            np.mean(band_mag + eps, axis=0)
        )
    else:
        flatness = np.ones(source_mag.shape[1], dtype=np.float32)
    flatness_gate = np.clip((flatness - 0.025) / 0.30, 0.0, 1.0)

    log_energy = np.log1p(np.mean(source_mag, axis=0))
    flux = np.maximum(0.0, np.diff(log_energy, prepend=log_energy[:1]))
    flux_scale = float(np.percentile(flux, 90)) if flux.size else 0.0
    onset_gate = np.clip(flux / max(flux_scale, 1e-5), 0.0, 1.0)

    frame_voicing = _frame_voicing(voiced, source_mag.shape[1])
    if frame_voicing is None:
        unvoiced_gate = flatness_gate
    else:
        unvoiced_gate = 1.0 - np.clip(frame_voicing, 0.0, 1.0)
    event_gate = np.clip(0.15 + 0.70 * flatness_gate + 0.15 * onset_gate, 0.0, 1.0)
    frame_gate = (0.025 + 0.975 * unvoiced_gate) * event_gate
    mask = amount * detail_band[:, None] * frame_gate[None, :]

    mixed_spec = generated_spec * (1.0 - mask) + recolored_source * mask
    mixed = librosa.istft(
        mixed_spec, hop_length=hop_length, win_length=n_fft, length=converted.size
    )
    return np.nan_to_num(mixed, copy=False).astype(np.float32)
