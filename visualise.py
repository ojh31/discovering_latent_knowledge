#%%
import argparse
import json
import pandas as pd
import plotly.express as px

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-path', type=str, default='eval.json')
    args, _ = parser.parse_known_args()

    with open(args.eval_path, 'r') as f:
        eval_d = json.load(f)
# %%
df = pd.Series(eval_d).reset_index()
df.columns = ['full_name', 'accuracy']
cols_to_extract = [
    'model_name', 'dataset_name', 'partition', 'reg', 
    'normalisation', 'weight_decay',
    'hidden_size', 'nepochs', 'ntries',
]
for col in cols_to_extract:
    df[col] = df.full_name.str.extract(f'{col}_((_?[^_]+)*)')[0]
df.weight_decay = df.weight_decay.astype(float)
df['reg_key'] = (
    df.reg + 
    '_nml_' +
    df.normalisation + 
    '_hs_' + 
    df.hidden_size +
    '_epochs_' +
    df.nepochs +
    '_tries_' + 
    df.ntries +
    '_wd_' +
    df.weight_decay.map(lambda x: f'{x:.04f}')
)
df.head()
#%%
[col for col in df.columns if (df[col] != df[col][0]).any()]
# %%
# Train vs. test
fig = px.line(
    df, x='partition', y='accuracy', color='reg_key', 
    facet_col='dataset_name', facet_row='model_name'
)
fig.show()

# %%
