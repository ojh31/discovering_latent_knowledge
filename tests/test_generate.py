import numpy as np
import os
import unittest as ut
from dlk.generate import main


def test_generated_file(filename: str):
    ref = np.load(os.path.join('reference_hidden_states', filename))
    act = np.load(os.path.join('generated_hidden_states', filename))
    np.testing.assert_allclose(
        ref, act, atol=5e-5, rtol=1e-4,
    )


class TestGenerate(ut.TestCase):

    def setUp(self):
        main([
            "--device",
            "cpu",
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
        self.filenames = os.listdir('reference_hidden_states')

    def test_labels(self):
        test_generated_file(self.filenames[0])

    def test_negatives(self):
        test_generated_file(self.filenames[1])

    def test_positives(self):
        test_generated_file(self.filenames[2])


if __name__ == '__main__':
    ut.main()
