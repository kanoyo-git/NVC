import unittest

import numpy as np

from infer.vc.pipeline import preserve_unvoiced_f0 as preserve_inference_f0
from train.f0_utils import preserve_unvoiced_f0 as preserve_training_f0


class UnvoicedF0Tests(unittest.TestCase):
    def test_training_keeps_gaps_and_tail_unvoiced(self):
        source = np.asarray([200.0, 0.0, 0.0, 220.0, 0.0], dtype=np.float32)
        result = preserve_training_f0(source)
        np.testing.assert_array_equal(result, source)

    def test_inference_sanitizes_invalid_values_without_filling_gaps(self):
        source = np.asarray([200.0, np.nan, -5.0, np.inf, 220.0], dtype=np.float32)
        result = preserve_inference_f0(source)
        np.testing.assert_array_equal(
            result,
            np.asarray([200.0, 0.0, 0.0, 0.0, 220.0], dtype=np.float32),
        )

    def test_helper_does_not_modify_the_callers_array(self):
        source = np.asarray([180.0, np.nan, 0.0], dtype=np.float32)
        preserve_training_f0(source)
        self.assertTrue(np.isnan(source[1]))


if __name__ == "__main__":
    unittest.main()
