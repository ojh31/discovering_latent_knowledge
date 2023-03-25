#%%
import json
import pandas as pd
import plotly.express as px
#%%
EVAL_PATH = 'results.json'
with open(EVAL_PATH, 'r') as f:
        eval_d = json.load(f)
    
# %%
df = pd.Series(eval_d).reset_index()
df.columns = ['full_name', 'accuracy']
bool_cols = [
    'mean_normalize', 'var_normalize','all_layers'
]
float_cols = [
    'weight_decay',
]
int_cols = [
      'hidden_size', 'nepochs', 'ntries',
]
str_cols = [
    'model_name', 'dataset_name', 'partition', 'reg', 
]
cols_to_extract = bool_cols + float_cols + int_cols + str_cols
for col in cols_to_extract:
    df[col] = df.full_name.str.extract(f'{col}_((_?[^_]+)*)')[0]
    if col in bool_cols:
        df[col] = df[col].map({'True': True, 'False': False})
df = df.astype({col: float for col in float_cols})
df['reg_key'] = (
    df.reg +
    '_layers_' +
    df.all_layers.map({True: 'all', False: 'last'})
    # '_nml_' +
    # df.mean_normalize.map({True: 'm', False: ''}) +
    # df.var_normalize.map({True: 'v', False: ''}) +
    # '_hs_' + 
    # df.hidden_size +
    # '_epochs_' +
    # df.nepochs +
    # '_tries_' + 
    # df.ntries +
    # '_wd_' +
    # df.weight_decay.map(lambda x: f'{x:.04f}')
)
df.head()
# %%
# Train vs. test
fig = px.line(
    df, x='partition', y='accuracy', color='reg_key', 
    facet_col='dataset_name', facet_row='model_name'
)
# for annotation in fig['layout']['annotations']: 
#     annotation['textangle']= 0
fig.show()

# %%
