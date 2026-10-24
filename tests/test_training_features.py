import json
import unittest
from pathlib import Path

from tools.pretrained_catalog import SAMPLE_RATES, load_catalog, target_paths


ROOT = Path(__file__).resolve().parent.parent


class PretrainedCatalogTests(unittest.TestCase):
    def test_entries_have_generator_and_discriminator(self):
        catalog = load_catalog()
        self.assertIn("RVC v2", catalog)
        for name, rates in catalog.items():
            self.assertTrue(rates, name)
            for sr, files in rates.items():
                self.assertIn(sr, SAMPLE_RATES, name)
                self.assertEqual(set(files), {"G", "D"}, name)
                self.assertIn("/resolve/", files["G"], name)
                self.assertIn("/resolve/", files["D"], name)

    def test_install_paths_match_training_tab_names(self):
        generator, discriminator = target_paths("RVC v2", "40k")
        self.assertEqual(generator.as_posix(), "assets/pretrained_v2/f0G40k.pth")
        self.assertEqual(discriminator.as_posix(), "assets/pretrained_v2/f0D40k.pth")
        generator, _ = target_paths("Snowie-X-RinE3", "40k")
        self.assertEqual(generator.as_posix(), "assets/pretrained_v2/Snowie-X-RinE3/f0G40k.pth")


class ColabNotebookTests(unittest.TestCase):
    def test_notebooks_drop_nvc_tunnel_and_add_training_cells(self):
        for name in ("NVC.ipynb", "NVC.ru.ipynb"):
            text = (ROOT / name).read_text(encoding="utf-8")
            notebook = json.loads(text)
            for gone in ("self_hosted", "frpc", "kanoyo.qzz.io"):
                self.assertNotIn(gone, text, name)
            ids = {cell["metadata"].get("id") for cell in notebook["cells"]}
            self.assertTrue({"NVC_DRIVE_TRAINING", "NVC_TENSORBOARD"} <= ids, name)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), name, "exec")


if __name__ == "__main__":
    unittest.main()
