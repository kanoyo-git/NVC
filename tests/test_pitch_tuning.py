import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from infer.vc.pitch_tuning import (
    PitchProfile,
    AdaptivePitchSmoother,
    hz_to_midi,
    load_pitch_profile,
    midi_to_hz,
    segment_vocal_phrases,
    select_dynamic_register_shifts,
    select_register_shift,
    stabilize_f0,
)
from tools.build_pitch_profile import StreamingPitchStats, collect_audio_files


class PitchTuningTests(unittest.TestCase):
    def setUp(self):
        self.profile = PitchProfile(
            name="Lover",
            quantiles={
                0.05: 48.99,
                0.10: 50.96,
                0.25: 54.86,
                0.50: 59.02,
                0.75: 62.29,
                0.90: 64.95,
                0.95: 66.03,
            },
        )

    def test_frequency_conversion_round_trip(self):
        frequencies = np.asarray((55.0, 220.0, 440.0, 880.0))
        np.testing.assert_allclose(
            midi_to_hz(hz_to_midi(frequencies)), frequencies, rtol=1e-12
        )

    def test_stabilizer_removes_single_octave_error(self):
        midi = np.full(80, 57.2)
        midi[35] += 12.0
        f0 = midi_to_hz(midi)
        stable = hz_to_midi(stabilize_f0(f0))
        self.assertLess(abs(stable[35] - 57.2), 0.01)
        self.assertLess(np.max(np.abs(np.diff(stable))), 0.02)

    def test_tuner_preserves_vibrato_and_unvoiced_frames(self):
        phase = np.arange(200) / 100.0
        source_midi = 57.28 + 0.22 * np.sin(2 * np.pi * 5.0 * phase)
        source_f0 = midi_to_hz(source_midi)
        source_f0[:8] = 0.0
        source_f0[-8:] = 0.0
        tuned = AdaptivePitchSmoother().tune(source_f0)
        tuned_midi = hz_to_midi(tuned[8:-8])
        self.assertTrue(np.all(tuned[:8] == 0.0))
        self.assertTrue(np.all(tuned[-8:] == 0.0))
        self.assertLess(abs(np.median(tuned_midi) - 57.0), 0.08)
        self.assertGreater(np.std(tuned_midi), np.std(source_midi[8:-8]) * 0.70)

    def test_tuner_keeps_a_smooth_note_transition(self):
        first = np.full(70, 57.22)
        glide = np.linspace(57.22, 59.18, 24)
        second = np.full(70, 59.18)
        source = midi_to_hz(np.concatenate((first, glide, second)))
        tuned = hz_to_midi(AdaptivePitchSmoother().tune(source))
        self.assertLess(np.max(np.abs(np.diff(tuned))), 0.20)
        self.assertLess(abs(np.median(tuned[:60]) - 57.0), 0.10)
        self.assertLess(abs(np.median(tuned[-60:]) - 59.0), 0.10)

    def test_register_match_uses_octaves_and_model_distribution(self):
        target = self.profile.values_at()
        low_source = midi_to_hz(np.repeat(target - 12.0, 30))
        high_source = midi_to_hz(np.repeat(target + 12.0, 30))
        low_shift, _ = select_register_shift(low_source, self.profile)
        high_shift, _ = select_register_shift(high_source, self.profile)
        self.assertEqual(low_shift, 12)
        self.assertEqual(high_shift, -12)
        self.assertEqual(low_shift % 12, 0)
        self.assertEqual(high_shift % 12, 0)

    def test_register_match_does_not_change_an_ambiguous_register(self):
        source = midi_to_hz(np.repeat(self.profile.values_at() + 5.8, 30))
        shift, _ = select_register_shift(source, self.profile)
        self.assertEqual(shift, 0)

    def test_dynamic_register_tracks_alternating_vocal_phrases(self):
        target = self.profile.values_at()
        low_phrase = midi_to_hz(np.tile(target - 12.0, 35))
        high_phrase = midi_to_hz(np.tile(target + 12.0, 35))
        gap = np.zeros(60)
        f0 = np.concatenate((gap, low_phrase, gap, high_phrase, gap))
        curve, report = select_dynamic_register_shifts(f0, self.profile)
        phrases = segment_vocal_phrases(f0)
        self.assertEqual(len(phrases), 2)
        self.assertEqual(len(curve), len(f0))
        self.assertEqual(report.phrases[0].shift, 12)
        self.assertEqual(report.phrases[1].shift, -12)
        self.assertTrue(np.all(curve[phrases[0][0] : phrases[0][1]] == 12))
        self.assertTrue(np.all(curve[phrases[1][0] : phrases[1][1]] == -12))
        self.assertTrue(np.all(curve[len(gap) + len(low_phrase) :][:20] == 0))

    def test_dynamic_register_ignores_an_isolated_short_extreme_phrase(self):
        target = self.profile.values_at()
        normal = midi_to_hz(np.tile(target, 30))
        short_low = midi_to_hz(np.repeat(target[0] - 12.0, 25))
        gap = np.zeros(20)
        f0 = np.concatenate((normal, gap, short_low, gap, normal))
        _, report = select_dynamic_register_shifts(f0, self.profile)
        self.assertEqual([phrase.shift for phrase in report.phrases], [0, 0, 0])

    def test_dynamic_register_never_switches_inside_a_voiced_phrase(self):
        target = self.profile.values_at()
        low = midi_to_hz(np.tile(target - 12.0, 20))
        high = midi_to_hz(np.tile(target + 12.0, 20))
        f0 = np.concatenate((low, high))
        curve, report = select_dynamic_register_shifts(f0, self.profile)
        self.assertEqual(len(report.phrases), 1)
        self.assertEqual(np.unique(curve).size, 1)

    def test_loads_packaged_model_profile(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model_path = root / "voice.pth"
            model_path.touch()
            payload = {
                "name": "test",
                "voiced_frames": 100,
                "midi_quantiles": {"0.1": 50, "0.5": 59, "0.9": 66},
            }
            (root / "pitch-profile.json").write_text(json.dumps(payload))
            profile, source = load_pitch_profile(model_path)
        self.assertEqual(profile.name, "test")
        self.assertEqual(profile.voiced_frames, 100)
        self.assertTrue(source.endswith("pitch-profile.json"))

    def test_dataset_profile_builder_scans_nested_audio(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            nested = root / "singer"
            nested.mkdir()
            (root / "a.wav").touch()
            (nested / "b.FLAC").touch()
            (nested / "ignore.txt").touch()
            source, files = collect_audio_files(root)
        self.assertEqual(source, root)
        self.assertEqual([path.name for path in files], ["a.wav", "b.FLAC"])

    def test_streaming_profile_statistics_match_pitch_range(self):
        stats = StreamingPitchStats()
        midi = np.linspace(48.0, 68.0, 10000)
        stats.update(midi_to_hz(midi[:5000]))
        stats.update(midi_to_hz(midi[5000:]))
        values = stats.quantiles((0.05, 0.5, 0.95))
        np.testing.assert_allclose(values, (49.0, 58.0, 67.0), atol=0.03)
        self.assertAlmostEqual(stats.mean, 58.0, places=2)


if __name__ == "__main__":
    unittest.main()
