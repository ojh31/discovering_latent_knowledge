#%%
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
from typeguard import typechecked
from typing import Union
import wandb
from utils import (
    get_parser, load_all_generations, args_to_filename, LatentKnowledgeMethod, MLPProbe
)
#%%
MAIN = __name__ == "__main__"
RUNNING_FROM_IPYNB = "ipykernel_launcher" in os.path.basename(sys.argv[0])
#%%
def clean_name(s: str):
    return s.lower().replace('-', '_')
#%%
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
    
#%%
@typechecked
def save_eval(
    reg: str, partition: str, val: float, args: Union[argparse.Namespace, dict]
):
    """
    Input: 
        key: name of field to write
        val: value corresponding to key
        args: cmd arguments used to generate

    Saves the evaluations to the eval file.
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if args.get('verbose_eval'):
        print(f'Setting {key}={val} for model={args.model_name}')
    key = args_to_filename(args) + f'__reg_{reg}__partition_{partition}'
    if os.path.isfile(args.get('eval_path')):
        with open(args.get('eval_path'), 'r') as f:
            eval_d = json.load(f)
    else:
        eval_d = dict()
    eval_d[key] = val
    with open(args.get('eval_path'), 'w') as f:
        f.write(json.dumps(eval_d))

#%%
def get_eval_args():
    parser = get_parser()
    generation_args, _ = parser.parse_known_args() 
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--normalisation", type=str, default='mean_and_var')
    parser.add_argument('--eval_path', type=str, default='eval.json')
    parser.add_argument('--verbose_eval', action='store_true')
    parser.add_argument('--plot_dir', type=str, default='plots')
    args_to_parse = [] if RUNNING_FROM_IPYNB else None
    args = parser.parse_args(args_to_parse)
    return args, generation_args

#%%
args, generation_args = get_eval_args()
arg_dict = vars(args)
#%%
# load hidden states and labels
neg_hs, pos_hs, y = load_all_generations(generation_args)
#%%
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

#%%
(
    neg_hs_train, neg_hs_test, 
    pos_hs_train, pos_hs_test, 
    y_train, y_test
) = split_train_test(
    neg_hs=neg_hs, pos_hs=pos_hs, y=y
)

#%% [markdown]
#### Logistic Regression
#%%
def fit_lr(
    neg_hs_train, neg_hs_test, pos_hs_train, pos_hs_test, y_train, y_test
):
    x_train = neg_hs_train - pos_hs_train  
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    save_eval('lr', 'train', lr.score(x_train, y_train), args)
    save_eval('lr', 'test', lr.score(x_test, y_test), args)
    lr_fi = (x_train.std(0) * lr.coef_).squeeze()
    plot_feature_importance(lr_fi, 'LR', args)
#%%
fit_lr(
    neg_hs_train=neg_hs_train, neg_hs_test=neg_hs_test, 
    pos_hs_train=pos_hs_train, pos_hs_test=pos_hs_test, 
    y_train=y_train, y_test=y_test,
)

#%% [markdown]
#### CCS
#%%

class CCS(LatentKnowledgeMethod):
    def __init__(
        self, 
        neg_hs_train, pos_hs_train, y_train,
        neg_hs_test, pos_hs_test, y_test,
        nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
        verbose=False, device="cuda", hidden_size=None, 
        weight_decay=0.01, mean_normalize=True,
        var_normalize=False,
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
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
                train_acc = self.get_train_acc() 
                test_acc = self.get_test_acc()
                wandb.log({
                    'train_accuracy': train_acc, 
                    'test_accuracy': test_acc
                }, step=train_num)
            wandb.log({'loss': loss}, step=train_num)

        return best_loss
#%%
def fit_ccs(
    neg_hs_train, neg_hs_test, pos_hs_train, 
    pos_hs_test, y_train, y_test, args
):
    wandb.init(config=args)
    config = wandb.config
    config_args = {
        k: config.get(k, v)
        for k, v in arg_dict.items()
    }
    mean_normalize = 'mean' in config.normalisation
    var_normalize = 'var' in config.normalisation
    ccs = CCS(
        neg_hs_train=neg_hs_train, pos_hs_train=pos_hs_train,
        y_train=y_train,
        neg_hs_test=neg_hs_test, pos_hs_test=pos_hs_test,
        y_test=y_test,
        nepochs=config.nepochs, ntries=config.ntries, 
        lr=config.lr, batch_size=config.ccs_batch_size, 
        verbose=config.verbose, device=config.ccs_device, 
        hidden_size=config.hidden_size,
        weight_decay=config.weight_decay, 
        mean_normalize=mean_normalize,
        var_normalize=var_normalize,
    )
    # train
    t0_train = time.time()
    ccs.repeated_train()
    print(f'Training completed in {time.time() - t0_train:.1f}s')
    # evaluate
    ccs_train_acc = ccs.get_train_acc()
    save_eval('ccs', 'train', ccs_train_acc, config_args)
    ccs_test_acc = ccs.get_test_acc()
    save_eval('ccs', 'test', ccs_test_acc, config_args)
    # feature importance
    if ccs.linear:
        ccs_fi = (ccs.best_probe[0].weight * ccs.pos_hs_train.std()).squeeze()
        plot_feature_importance(ccs_fi, 'CCS', config)
# %%
fit_ccs(
    neg_hs_train=neg_hs_train, neg_hs_test=neg_hs_test, 
    pos_hs_train=pos_hs_train, pos_hs_test=pos_hs_test, 
    y_train=y_train, y_test=y_test,
    args=args,
)
#%%
