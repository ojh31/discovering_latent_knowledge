#%%
import os
import sys
import lightgbm as lgb
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS
#%%
MAIN = __name__ == "__main__"
RUNNING_FROM_IPYNB = "ipykernel_launcher" in os.path.basename(sys.argv[0])

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
if RUNNING_FROM_IPYNB:
    args, _ = parser.parse_known_args()
    args.model_name = generation_args.model_name = 'deberta-l'
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

#%%
#### Logistic Regression
# Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
# you can also concatenate, but this works fine and is more comparable to CCS inputs
x_train = neg_hs_train - pos_hs_train  
x_test = neg_hs_test - pos_hs_test
lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)
print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))

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
#%%
#### LGBM
# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# specify your configurations as a dict
lgb_params = {
    'objective': 'binary',
    'metric': 'binary',
    'max_depth': 2,
    'num_leaves': 3,
    'learning_rate': 0.05,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 2,
    'verbose': 0,
}

gbm = lgb.train(lgb_params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=5)])

# Train accuracy
gbm_train_pred = gbm.predict(x_train, num_iteration=gbm.best_iteration).round().astype(int) 
gbm_train_acc = float((gbm_train_pred == y_train).sum()) / len(y_train)
print("Decision tree fit accuracy: {}".format(gbm_train_acc))

# Test accuracy
gbm_test_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration).round().astype(int) 
gbm_test_acc = float((gbm_test_pred == y_test).sum()) / len(y_test)
print("Decision tree test accuracy: {}".format(gbm_test_acc))

#%%
# Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                var_normalize=args.var_normalize)

# train and evaluate CCS
ccs.repeated_train()
ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
print("CCS accuracy: {}".format(ccs_acc))
