import argparse
import numpy as np
import unittest as ut
from dlk.evaluate import run_eval

GEN_ARGS = argparse.Namespace(
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
    save_dir='reference_hidden_states',
)
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
    save_dir='reference_hidden_states',
    cache_dir=None,

    ccs_batch_size=-1,
    ccs_device='cpu',
    hidden_size=0,
    lr=1e-3,
    weight_decay=.01,
    nepochs=1000,
    ntries=1,
    mean_normalize=True,
    var_normalize=True,
    wandb_enabled=False,
    eval_path=None,
    plot_dir=None,
    verbose=True,
    lr_max_iter=100,
    lr_solver='lbfgs',
    lr_inv_reg=1.0,
)


class TestEvaluate(ut.TestCase):

    def setUp(self) -> None:
        (
            self.lr_train_acc, self.lr_test_acc,
            self.ccs_train_acc, self.ccs_test_acc,
        ) = run_eval(GEN_ARGS, ARGS)

    def testLRTrainAcc(self):
        np.testing.assert_allclose(
            self.lr_train_acc,
            1.0
        )

    def testLRTestAcc(self):
        np.testing.assert_allclose(
            self.lr_test_acc,
            0.4
        )

    def testCCSTrainAcc(self):
        np.testing.assert_allclose(
            self.ccs_train_acc,
            0.8
        )

    def testCCSTestAcc(self):
        np.testing.assert_allclose(
            self.ccs_train_acc,
            0.8
        )
