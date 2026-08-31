import unittest

import torch

from train.losses import MultiResolutionSTFTLoss


class SpectralDetailLossTests(unittest.TestCase):
    def setUp(self):
        self.loss = MultiResolutionSTFTLoss(
            resolutions=((128, 32, 128), (256, 64, 256))
        )

    def test_identical_audio_has_zero_loss(self):
        audio = torch.randn(2, 1, 1024)
        self.assertAlmostEqual(float(self.loss(audio, audio)), 0.0, places=6)

    def test_loss_is_differentiable(self):
        real = torch.randn(1, 1, 1024)
        generated = torch.randn(1, 1, 1024, requires_grad=True)
        value = self.loss(real, generated)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertIsNotNone(generated.grad)
        self.assertTrue(torch.isfinite(generated.grad).all())


if __name__ == "__main__":
    unittest.main()
