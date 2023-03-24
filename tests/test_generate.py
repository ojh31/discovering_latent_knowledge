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
    FILENAMES = [
        'negative_hidden_states__model_name_deberta-l__parallelize_False__dataset_name_truthful_qa__split_test__config_name_multiple_choice__prompt_idx_0__batch_size_1__num_examples_10__seed_0__use_decoder_False__layer_-1__all_layers_False__token_idx_-1.npy', 
        'positive_hidden_states__model_name_deberta-l__parallelize_False__dataset_name_truthful_qa__split_test__config_name_multiple_choice__prompt_idx_0__batch_size_1__num_examples_10__seed_0__use_decoder_False__layer_-1__all_layers_False__token_idx_-1.npy', 
        'labels__model_name_deberta-l__parallelize_False__dataset_name_truthful_qa__split_test__config_name_multiple_choice__prompt_idx_0__batch_size_1__num_examples_10__seed_0__use_decoder_False__layer_-1__all_layers_False__token_idx_-1.npy'
    ]

    @classmethod
    def setUpClass(cls):
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

    def test_negatives(self):
        test_generated_file(self.FILENAMES[0])

    def test_positives(self):
        test_generated_file(self.FILENAMES[1])

    def test_labels(self):
        test_generated_file(self.FILENAMES[2])


if __name__ == '__main__':
    ut.main()
