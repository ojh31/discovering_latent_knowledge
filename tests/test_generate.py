import numpy as np
import os
import unittest as ut
from dlk.generate import main


class TestGenerate(ut.TestCase):

    def setUp(self):
        main([
            "--device",
            "cuda",
            "--model_name",
            "deberta-l",
            "--dataset_name",
            "truthful_qa",
            "--config_name",
            "multiple_choice",
            "--num_examples",
            "10",
            "--prompt_idx",
            "0",
        ])

    def test_generated(self):
        for filename in os.listdir('reference_hidden_states'):
            ref = np.load(os.path.join('reference_hidden_states', filename))
            act = np.load(os.path.join('generated_hidden_states', filename))
            np.testing.assert_allclose(ref, act)


if __name__ == '__main__':
    ut.main()
