#%%
import json
import os
import sys
import time
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS
#%%
MAIN = __name__ == "__main__"
RUNNING_FROM_IPYNB = "ipykernel_launcher" in os.path.basename(sys.argv[0])


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
    key = args.model_name + '__' + key
    if os.path.isfile(args.eval_path):
        with open(args.eval_path, 'r') as f:
            eval_d = json.load(f)
    else:
        eval_d = dict()
    eval_d[key] = val
    with open(args.eval_path, 'w') as f:
        f.write(json.dumps(eval_d))


def main(args, generation_args):
    pass


#%%
if MAIN:
    parser = get_parser()
    generation_args, _ = parser.parse_known_args() # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    parser.add_argument('--eval_path', type=str, default='eval.json')
    parser.add_argument('--verbose-eval', action='store_true')
if RUNNING_FROM_IPYNB:
    args, _ = parser.parse_known_args()
    args.model_name = generation_args.model_name = 'deberta-l'
    args.verbose_eval = True
else:
    args = parser.parse_args()
    # main(args, generation_args)

    
#%%
# load hidden states and labels
neg_hs, pos_hs, y = load_all_generations(generation_args)

#%%
# Make sure the shape is correct
assert neg_hs.shape == pos_hs.shape
neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
    neg_hs = neg_hs.squeeze(1)
    pos_hs = pos_hs.squeeze(1)
#%%
# Very simple train/test split (using the fact that the data is already shuffled)
neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]

#%% [markdown]
#### Logistic Regression
#%%
# Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
# you can also concatenate, but this works fine and is more comparable to CCS inputs
x_train = neg_hs_train - pos_hs_train  
x_test = neg_hs_test - pos_hs_test
lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)
save_eval('lr_train_acc', lr.score(x_train, y_train), args)
save_eval('lr_test_acc', lr.score(x_test, y_test), args)

#%%
#### LR feature importance
lr_fi = (x_train.std(0) * lr.coef_).squeeze()
fig = px.bar(
    y=lr_fi, 
    title='LR feature importance',
    labels={'y': 'importance', 'x': 'feature'},
)
fig.update_layout({
    'title_x': 0.5
})
fig.show()
#%%
fig = px.histogram(
    x=lr_fi, 
    title='LR feature importance distribution',
)
fig.update_layout({
    'title_x': 0.5,
})
fig.show()

#%% [markdown]
#### CCS

#%%
ccs = CCS(
    neg_hs_train, pos_hs_train, 
    nepochs=args.nepochs, ntries=args.ntries, 
    lr=args.lr, batch_size=args.ccs_batch_size, 
    verbose=args.verbose, device=args.ccs_device, 
    linear=args.linear, 
    hidden_size=args.hidden_size,
    weight_decay=args.weight_decay, 
    var_normalize=args.var_normalize
)
#%%
# train
t0_train = time.time()
ccs.repeated_train()
print(f'Training completed in {time.time() - t0_train:.1f}s')
#%%
# evaluate
t0_acc = time.time()
ccs_train_acc = ccs.get_acc(neg_hs_train, pos_hs_train, y_train)
save_eval('ccs_train_acc', ccs_train_acc, args)
ccs_test_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
save_eval('ccs_test_acc', ccs_test_acc, args)
print(f'Accuracy computed in {time.time() - t0_acc:.1f}s')

# %%
# TODO:
# * compute SHAP score for best_probe
# * search model/data pairs
# * relationship between spread and model size or type?
# * find a model/data pair with a large LR/CCS spread then
#   * sweep hyperparameter sweeps
#   * experiment with normalisation
#   * experiment with transfer learning
# * look at examples where final-layer and all-layer strongly disagree
