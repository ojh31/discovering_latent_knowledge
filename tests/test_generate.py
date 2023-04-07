import argparse
import numpy as np
import os
import unittest as ut
from dlk.generate import run_gen

ARGS = argparse.Namespace(
    device='cpu',
    model_name='deberta-l',
    dataset_name='truthful_qa/multiple_choice',
    split='test',
    num_examples=10,
    prompt_idx=0,
    parallelize=False,
    batch_size=1,
    seed=0,
    use_decoder=False,
    layer=-1,
    all_layers=False,
    token_idx=-1,
    save_dir='generated_hidden_states',
    cache_dir=None,
    verbose=False,
)
FILENAMES = [
    'negative_hidden_states__all_layers_False__batch_size_1__dataset_name_truthful_qa_multiple_choice__layer_-1__model_name_deberta-l__num_examples_10__parallelize_False__prompt_idx_0__seed_0__split_test__token_idx_-1__use_decoder_False.npy',
    'positive_hidden_states__all_layers_False__batch_size_1__dataset_name_truthful_qa_multiple_choice__layer_-1__model_name_deberta-l__num_examples_10__parallelize_False__prompt_idx_0__seed_0__split_test__token_idx_-1__use_decoder_False.npy', 
    'labels__all_layers_False__batch_size_1__dataset_name_truthful_qa_multiple_choice__layer_-1__model_name_deberta-l__num_examples_10__parallelize_False__prompt_idx_0__seed_0__split_test__token_idx_-1__use_decoder_False.npy', 
]
REF_ROOT = 'reference_hidden_states'
GEN_ROOT = 'generated_hidden_states'


def clear_cached_files():
    for filename in FILENAMES:
        path = os.path.join(GEN_ROOT, filename)
        if os.path.exists(path):
            os.remove(path)


def test_generated_file(filename: str):
    ref = np.load(os.path.join(REF_ROOT, filename))
    act = np.load(os.path.join(GEN_ROOT, filename))
    np.testing.assert_allclose(
        ref, act, atol=5e-5, rtol=1e-4,
    )


class TestGenerate(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        clear_cached_files()
        run_gen(ARGS)

    def test_negatives(self):
        test_generated_file(FILENAMES[0])

    def test_positives(self):
        test_generated_file(FILENAMES[1])

    def test_labels(self):
        test_generated_file(FILENAMES[2])


if __name__ == '__main__':
    ut.main()
