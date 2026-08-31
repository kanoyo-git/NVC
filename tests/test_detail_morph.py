import unittest
from unittest import mock

import numpy as np

from infer.vc.detail_morph import adaptive_detail_morph
from infer.cli import build_parser


class DetailMorphTests(unittest.TestCase):
    def setUp(self):
        self.sr = 48000
        self.length = self.sr // 2
        self.time = np.arange(self.length, dtype=np.float32) / self.sr

    def test_zero_amount_is_bit_exact(self):
        converted = np.sin(2 * np.pi * 220 * self.time).astype(np.float32)
        source = np.random.default_rng(7).normal(0, 0.1, self.length).astype(np.float32)
        result = adaptive_detail_morph(converted, source, self.sr, 0.0)
        np.testing.assert_array_equal(result, converted)

    def test_cli_exposes_detail_mix_with_compatible_default(self):
        with mock.patch(
            "sys.argv", ["nvc", "--model", "voice.pth"]
        ):
            args = build_parser().parse_args()
        self.assertEqual(args.detail_mix, 0.0)

    def test_unvoiced_noise_detail_is_added_without_changing_length(self):
        rng = np.random.default_rng(11)
        converted = np.zeros(self.length, dtype=np.float32)
        source = rng.normal(0, 0.1, self.length).astype(np.float32)
        result = adaptive_detail_morph(
            converted,
            source,
            self.sr,
            0.10,
            voiced=np.zeros(50, dtype=np.float32),
        )
        self.assertEqual(result.shape, converted.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(np.sqrt(np.mean(result**2))), 1e-4)

    def test_voiced_tone_is_strongly_protected_from_source_pitch(self):
        converted = 0.2 * np.sin(2 * np.pi * 220 * self.time)
        source = 0.2 * np.sin(2 * np.pi * 330 * self.time)
        result = adaptive_detail_morph(
            converted,
            source,
            self.sr,
            0.20,
            voiced=np.ones(50, dtype=np.float32),
        )
        relative_change = np.sqrt(np.mean((result - converted) ** 2)) / np.sqrt(
            np.mean(converted**2)
        )
        self.assertLess(float(relative_change), 0.01)


if __name__ == "__main__":
    unittest.main()
