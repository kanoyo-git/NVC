import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from infer import hubert


ROOT = Path(__file__).resolve().parent.parent


class MHubert147IntegrationTests(unittest.TestCase):
    def test_registered_with_official_safe_weights(self):
        self.assertIn("mhubert147", hubert.EMBEDDER_CHOICES)
        self.assertIn("mhubert147-phonetic", hubert.EMBEDDER_CHOICES)
        spec = hubert.EMBEDDER_DOWNLOAD_SPECS["mhubert147"]
        self.assertEqual(
            spec["base_url"],
            "https://huggingface.co/utter-project/mHuBERT-147/resolve/main",
        )
        self.assertEqual(
            spec["files"],
            ("model.safetensors", "config.json", "preprocessor_config.json"),
        )

    def test_download_uses_mhubert_repository(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = [b"model", b""] * 3
        with tempfile.TemporaryDirectory() as root, mock.patch(
            "infer.hubert.urllib.request.urlopen", return_value=response
        ) as urlopen:
            model_dir = Path(root) / "mhubert147"
            hubert._download_embedder(model_dir, "mhubert147")

            self.assertEqual(urlopen.call_count, 3)
            requested = [call.args[0] for call in urlopen.call_args_list]
            self.assertTrue(all("utter-project/mHuBERT-147" in url for url in requested))
            self.assertTrue((model_dir / "model.safetensors").is_file())

    def test_phonetic_variant_reuses_mhubert_weights(self):
        self.assertEqual(
            hubert.embedder_directory("mhubert147-phonetic"),
            hubert.embedder_directory("mhubert147"),
        )

    def test_phonetic_variant_mixes_intermediate_layers(self):
        class LayerModel(torch.nn.Module):
            _nvc_audio_requires_normalization = False
            _nvc_layer_mix = hubert.PHONETIC_LAYER_MIX

            def forward(self, input_values, **kwargs):
                hidden_states = tuple(
                    torch.full((*input_values.shape, 768), float(layer))
                    for layer in range(13)
                )
                return SimpleNamespace(hidden_states=hidden_states)

        features = hubert.extract_hubert_features(
            LayerModel(), torch.zeros(1, 4)
        )
        expected = sum(
            layer * weight for layer, weight in hubert.PHONETIC_LAYER_MIX
        )
        self.assertEqual(tuple(features.shape), (1, 4, 768))
        self.assertAlmostEqual(float(features.mean()), expected, places=5)

    def test_audio_normalization_is_shared_by_training_and_inference(self):
        class CapturingModel(torch.nn.Module):
            _nvc_embedder_name = "mhubert147"
            _nvc_audio_requires_normalization = True

            def forward(self, input_values, **kwargs):
                self.input_values = input_values
                features = input_values.unsqueeze(-1).expand(-1, -1, 768)
                return SimpleNamespace(last_hidden_state=features)

        model = CapturingModel()
        source = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        features = hubert.extract_hubert_features(model, source)

        self.assertEqual(tuple(features.shape), (1, 4, 768))
        self.assertAlmostEqual(float(model.input_values.mean()), 0.0, places=6)
        self.assertAlmostEqual(
            float(model.input_values.var(unbiased=False)), 1.0, places=5
        )

    def test_studio_and_notebooks_expose_mhubert(self):
        studio = (ROOT / "tools" / "studio_static" / "index.html").read_text()
        self.assertIn('<option value="mhubert147">mhubert147</option>', studio)
        self.assertIn(
            '<option value="mhubert147-phonetic">mhubert147-phonetic</option>',
            studio,
        )

        for name in ("NVC.ipynb", "NVC.ru.ipynb"):
            notebook = json.loads((ROOT / name).read_text())
            sources = "\n".join(
                "".join(cell.get("source", ())) for cell in notebook["cells"]
            )
            self.assertIn('MHUBERT_REPO = "utter-project/mHuBERT-147"', sources)
            self.assertIn("model.safetensors", sources)


if __name__ == "__main__":
    unittest.main()
