import numpy as np
import unittest as ut
from dlk.evaluate import main


class TestEvaluate(ut.TestCase):

    def setUp(self) -> None:
        accuracy = main([
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
            "--save_dir",
            "reference_hidden_states",

            "--ccs_batch_size",
            "-1",
            "--ccs_device",
            "cuda",
            "--hidden-size",
            "0",
            "--weight_decay",
            "0.01",
            "--nepochs",
            "1000",
            "--ntries",
            "1",
            "--mean_normalize",
            "--var_normalize",
            "--verbose-eval",
        ])
        (
            self.lr_train_acc, self.lr_test_acc,
            self.ccs_train_acc, self.ccs_test_acc,
        ) = accuracy

    def testLRTrainAcc(self):
        np.testing.assert_allclose(
            self.lr_train_acc,
            1.0
        )

    def testLRTestAcc(self):
        np.testing.assert_allclose(
            self.lr_test_acc,
            1.0
        )

    def testCCSTrainAcc(self):
        np.testing.assert_allclose(
            self.ccs_train_acc,
            0.6
        )

    def testCCSTestAcc(self):
        np.testing.assert_allclose(
            self.ccs_train_acc,
            0.6
        )
