from pathlib import Path
import unittest

import numpy as np

from infer.cli import build_parser
from infer.nmvpe import to_viterbi_f0


ROOT = Path(__file__).resolve().parent.parent


class NMVPEIntegrationTests(unittest.TestCase):
    def test_stable_decoder_rejects_one_frame_octave_jump(self):
        hidden = np.zeros((12, 360), dtype=np.float32)
        hidden[:, 100] = 0.9
        hidden[6, 100] = 0.85
        hidden[6, 160] = 0.95

        f0 = to_viterbi_f0(
            hidden,
            thred=0.025,
            jump_cost=0.09,
            voiced_switch_cost=1.0,
        )

        expected = 10.0 * 2.0 ** ((100 * 20.0 + 1997.3794084376191) / 1200.0)
        self.assertTrue(np.all(f0 > 0))
        self.assertLess(float(np.max(np.abs(f0 - expected))), 0.1)

    def test_cli_accepts_nmvpe(self):
        args = build_parser().parse_args(
            ["--model", "voice.pth", "--list-speakers", "--f0-method", "nmvpe"]
        )
        self.assertEqual(args.f0_method, "nmvpe")

    def test_nmvpe_is_exposed_only_in_studio_gui(self):
        studio = (ROOT / "tools" / "studio_static" / "index.html").read_text()
        realtime = (ROOT / "realtime_gui.py").read_text()

        self.assertEqual(studio.count('value="nmvpe"'), 3)
        self.assertEqual(studio.count("/> nmvpe</label>"), 3)
        self.assertNotIn('key="nmvpe"', realtime)


if __name__ == "__main__":
    unittest.main()
