import json
import tempfile
import unittest
from pathlib import Path

import torch

from infer.checkpoint import load_inference_checkpoint
from train.process_ckpt import require_v2_weights


ROOT = Path(__file__).resolve().parent.parent


class V2OnlyTests(unittest.TestCase):
    def _checkpoint(self, version="v2"):
        config = [0] * 18
        config[-2] = 256
        config[-1] = 40000
        return {
            "config": config,
            "weight": {
                "emb_g.weight": torch.zeros(1, 256),
                "enc_p.emb_phone.weight": torch.zeros(192, 768),
            },
            "version": version,
            "f0": 1,
        }

    def test_inference_accepts_only_v2_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save(self._checkpoint(), path)
            self.assertEqual(load_inference_checkpoint(path)["version"], "v2")

            torch.save(self._checkpoint("v1"), path)
            with self.assertRaisesRegex(ValueError, "only v2"):
                load_inference_checkpoint(path)

    def test_checkpoint_tools_verify_768_feature_width(self):
        require_v2_weights(self._checkpoint())
        checkpoint = self._checkpoint()
        checkpoint["weight"]["enc_p.emb_phone.weight"] = torch.zeros(192, 256)
        with self.assertRaises(ValueError):
            require_v2_weights(checkpoint)

    def test_old_configs_and_version_controls_are_removed(self):
        self.assertFalse((ROOT / "configs" / "v1").exists())
        self.assertTrue((ROOT / "configs" / "v2" / "40k.json").is_file())

        studio = (ROOT / "tools" / "studio_static" / "index.html").read_text()
        legacy = (ROOT / "legacy_gui.py").read_text()
        self.assertNotIn('value="v1"', studio)
        self.assertNotIn('choices=["v1", "v2"]', legacy)


class GuiSeparationTests(unittest.TestCase):
    def test_entrypoints_are_separate(self):
        studio_entrypoint = (ROOT / "gui.py").read_text()
        legacy_entrypoint = (ROOT / "legacy_gui.py").read_text()
        self.assertIn("launch_studio", studio_entrypoint)
        self.assertNotIn("import gradio", studio_entrypoint)
        self.assertNotIn("gr.Blocks", studio_entrypoint)
        self.assertIn("import gradio as gr", legacy_entrypoint)
        self.assertIn("gr.Blocks", legacy_entrypoint)

    def test_notebooks_select_the_entrypoint_file(self):
        for name in ("NVC.ipynb", "NVC.ru.ipynb"):
            notebook = json.loads((ROOT / name).read_text())
            source = "\n".join(
                "".join(cell.get("source", ())) for cell in notebook["cells"]
            )
            self.assertIn(
                'entrypoint = "legacy_gui.py" if INTERFACE == "legacy" else "gui.py"',
                source,
            )
            self.assertIn(
                'NVC_GIT_REF = "main"  # @param ["main", "experimental"]',
                source,
            )
            self.assertIn('"--branch", NVC_GIT_REF', source)
            self.assertNotIn('cmd.append("--legacy")', source)

    def test_theme_button_has_distinct_sun_and_moon_icons(self):
        html = (ROOT / "tools" / "studio_static" / "index.html").read_text()
        css = (ROOT / "tools" / "studio_static" / "app.css").read_text()
        self.assertIn("theme-icon-sun", html)
        self.assertIn("theme-icon-moon", html)
        self.assertIn('html[data-theme="light"] .theme-icon-sun', css)
        self.assertIn('html[data-theme="light"] .theme-icon-moon', css)


if __name__ == "__main__":
    unittest.main()
