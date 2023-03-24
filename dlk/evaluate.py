import argparse
import copy
import json
import numpy as np
import os
import sys
import time
import plotly.offline as off
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from typing import Union, List
import wandb
from dlk.utils import get_parser, load_all_generations, MLPProbe, LatentKnowledgeMethod


def clean_name(s: str):
    return s.lower().replace('-', '_')


def plot_feature_importance(
    vec: Union[np.ndarray, torch.Tensor], label: str, a: argparse.Namespace,
):
    model_name = clean_name(a.model_name)
    data_name = clean_name(a.dataset_name)
    label_clean = clean_name(label)
    if isinstance(vec, torch.Tensor):
        vec = vec.cpu().detach().numpy()
    fig = px.histogram(
        x=vec, 
        title=f'{label} feature importance distribution',
    )
    fig.update_layout({
        'title_x': 0.5,
    })
    if not os.path.exists(a.plot_dir):
        os.mkdir(a.plot_dir)
    off.plot(
        fig, 
        filename=os.path.join(
            a.plot_dir, 
            f'{model_name}_{data_name}_{label_clean}_feature_importance.html'
        ),
        auto_open=False,
    )
    

def save_eval(key, val, args):
    """
    Input: 
        key: name of field to write
        val: value corresponding to key
        args: cmd arguments used to generate

    Saves the evaluations to the eval file.
    """
    if args.verbose_eval:
        print(f'Setting {key}={val} for model={args.model_name}')
    key = args.model_name + '__' + args.dataset_name + '__' + key
    if os.path.isfile(args.eval_path):
        with open(args.eval_path, 'r') as f:
            eval_d = json.load(f)
    else:
        eval_d = dict()
    eval_d[key] = val
    with open(args.eval_path, 'w') as f:
        f.write(json.dumps(eval_d))


def parse_args(argv: List[str]):
    parser = get_parser()
    generation_args, _ = parser.parse_known_args(argv) # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mean_normalize", action=argparse.BooleanOptionalAction)
    parser.add_argument("--var_normalize", action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval_path', type=str, default='eval.json')
    parser.add_argument('--verbose_eval', action='store_true')
    parser.add_argument('--plot-dir', type=str, default='plots')
    args = parser.parse_args(argv)
    return generation_args, args


def split_train_test(neg_hs, pos_hs, y):
    # Make sure the shape is correct
    assert neg_hs.shape == pos_hs.shape
    neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        neg_hs = neg_hs.squeeze(1)
        pos_hs = pos_hs.squeeze(1)
    # Very simple train/test split (using the fact that the data is already shuffled)
    neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
    pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
    y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]
    return (
        neg_hs_train, neg_hs_test,
        pos_hs_train, pos_hs_test,
        y_train, y_test
    )

def fit_lr(
    neg_hs_train, pos_hs_train, neg_hs_test, pos_hs_test, y_train, y_test, args
):
    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs
    x_train = neg_hs_train - pos_hs_train  
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    lr_train_acc = lr.score(x_train, y_train)
    lr_test_acc = lr.score(x_test, y_test)
    save_eval('lr_train_acc', lr_train_acc, args)
    save_eval('lr_test_acc', lr_test_acc, args)
    lr_fi = (x_train.std(0) * lr.coef_).squeeze()
    plot_feature_importance(lr_fi, 'LR', args)
    return lr_train_acc, lr_test_acc


class CCS(LatentKnowledgeMethod):
    def __init__(
        self, 
        neg_hs_train: torch.Tensor, 
        pos_hs_train: torch.Tensor, 
        y_train: torch.Tensor,
        neg_hs_test: torch.Tensor, 
        pos_hs_test: torch.Tensor, 
        y_test: torch.Tensor,
        nepochs: int = 1000, 
        ntries: int = 10, 
        seed: int = 0, 
        lr: int = 1e-3, 
        batch_size: int = -1, 
        verbose: bool = False, 
        device: str = "cuda", 
        hidden_size: int = 0, 
        weight_decay: float = 0.01, 
        mean_normalize: bool = True,
        var_normalize: bool = False,
    ):
        super().__init__(
            neg_hs_train=neg_hs_train, 
            pos_hs_train=pos_hs_train, 
            y_train=y_train,
            neg_hs_test=neg_hs_test, 
            pos_hs_test=pos_hs_test, 
            y_test=y_test,
            mean_normalize=mean_normalize, 
            var_normalize=var_normalize,
            device=device,
        )

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        
        # probe
        self.hidden_size = hidden_size
        self.linear = hidden_size is None or (hidden_size == 0)
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d, self.hidden_size)
        self.probe.to(self.device)    
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        torch.manual_seed(self.seed)
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
                wandb.log({
                    'train_accuracy': self.get_train_acc(),
                    'test_accuracy': self.get_test_acc(),
                }, step=train_num)
            wandb.log({'loss': loss}, step=train_num)

        return best_loss



def fit_ccs(
    neg_hs_train, pos_hs_train, y_train,
    neg_hs_test, pos_hs_test, y_test,
    args,
):
    ccs = CCS(
        neg_hs_train=neg_hs_train, 
        pos_hs_train=pos_hs_train, 
        y_train=y_train,
        neg_hs_test=neg_hs_test, 
        pos_hs_test=pos_hs_test, 
        y_test=y_test,
        nepochs=args.nepochs, ntries=args.ntries, 
        lr=args.lr, batch_size=args.ccs_batch_size, 
        verbose=args.verbose, device=args.ccs_device, 
        hidden_size=args.hidden_size,
        weight_decay=args.weight_decay, 
        var_normalize=args.var_normalize
    )
    # train
    t0_train = time.time()
    ccs.repeated_train()
    print(f'Training completed in {time.time() - t0_train:.1f}s')
    ccs_train_acc = ccs.get_train_acc()
    save_eval('ccs_train_acc', ccs_train_acc, args)
    ccs_test_acc = ccs.get_test_acc()
    save_eval('ccs_test_acc', ccs_test_acc, args)

    if ccs.linear:
        ccs_fi = (ccs.best_probe[0].weight * ccs.pos_hs_train.std()).squeeze()
        plot_feature_importance(ccs_fi, 'CCS', args)
    return ccs_train_acc, ccs_test_acc


def main(argv: List[str]):
    generation_args, args = parse_args(argv)
    wandb.init(config=args)
    # load hidden states and labels
    neg_hs, pos_hs, y = load_all_generations(generation_args)
    (
        neg_hs_train, neg_hs_test,
        pos_hs_train, pos_hs_test,
        y_train, y_test
    ) = split_train_test(neg_hs, pos_hs, y)
    lr_train_acc, lr_test_acc = fit_lr(
        neg_hs_train=neg_hs_train,
        pos_hs_train=pos_hs_train,
        neg_hs_test=neg_hs_test,
        pos_hs_test=pos_hs_test,
        y_train=y_train,
        y_test=y_test,
        args=args,
    )
    ccs_train_acc, ccs_test_acc = fit_ccs(
        neg_hs_train=neg_hs_train,
        pos_hs_train=pos_hs_train,
        neg_hs_test=neg_hs_test,
        pos_hs_test=pos_hs_test,
        y_train=y_train,
        y_test=y_test,
        args=args,
    )
    wandb.finish()
    return (
        lr_train_acc, lr_test_acc,
        ccs_train_acc, ccs_test_acc,
    )


if __name__ == '__main__':
    main(sys.argv[1:])