#%%
import json
import pandas as pd
import plotly.express as px
import plotly.offline as off
#%%
EVAL_PATH = 'results.json'
PLOT_DIR = 'plots'
with open(EVAL_PATH, 'r') as f:
        eval_d = json.load(f)
    
# %%
df = pd.Series(
     {k: v for k, v in eval_d.items() if not isinstance(v, list)}
).reset_index()
df.columns = ['full_name', 'value']
bool_cols = [
    'mean_normalize', 'var_normalize','all_layers'
]
float_cols = [
    'weight_decay',
]
int_cols = [
      'hidden_size', 'nepochs', 'ntries', 'num_examples',
]
str_cols = [
    'model_name', 'dataset_name', 'partition', 'regression', 'kind',
    'split',
]
cols_to_extract = bool_cols + float_cols + int_cols + str_cols
for col in cols_to_extract:
    df[col] = df.full_name.str.extract(f'{col}_((_?[^_]+)*)')[0]
    if col in bool_cols:
        df[col] = df[col].map({'True': True, 'False': False})
df = df.astype({col: float for col in float_cols})
df = df.astype({col: int for col in int_cols})
df.dataset_name = df.dataset_name.str.replace('_multiple_choice', '')
df['reg_key'] = (
    df.regression +
    '_layers_' +
    df.all_layers.map({True: 'all', False: 'last'})
)
df.head()
#%%
# Confidence plots
conf_data = []
for k, v in eval_d.items():
    if not isinstance(v, list) or not 'conf' in k:
        continue
    k = k.replace('confidence', 'accuracy')
    c_df = pd.DataFrame({'conf': v, 'full_name': k})
    c_df = c_df.merge(df, how='left', on='full_name')
    c_df.kind = 'confidence'
    c_df.conf = c_df.conf.astype(float)
    conf_data.append(c_df)
conf_df = pd.concat(conf_data, ignore_index=True)
#%%
fig = px.histogram(
        conf_df, 
        x='conf', 
        color='reg_key',
        facet_col='dataset_name', 
        facet_row='model_name',
        title='Confidence',
)
fig.update_layout(title_x=0.5)
plot_path = off.plot(
    fig, 
    filename=f'{PLOT_DIR}/confidence_lr_vs_ccs.html',
    auto_open=True,
)
    
# %%
# Train vs. test
fig = px.line(
    df.loc[df.kind.eq('accuracy')].rename(columns={'value': 'accuracy'}), 
    x='partition', y='accuracy', color='reg_key', 
    facet_col='dataset_name', facet_row='model_name', title='Accuracy',
)
fig.update_layout(title_x=0.5)
plot_path = off.plot(
    fig, 
    filename=f'{PLOT_DIR}/accuracy_lr_vs_ccs.html',
    auto_open=True,
)

# %%
fig = px.line(
    df.loc[
        df.partition.eq('train') & 
        df.kind.ne('accuracy') &
        df.regression.eq('lr')
    ], 
    x='kind', y='value', color='reg_key', 
    facet_col='dataset_name', facet_row='model_name', 
    title='Samples vs. features',
    # barmode='overlay',
)
fig.update_layout(dict(title_x=0.5))
plot_path = off.plot(
    fig, 
    filename=f'{PLOT_DIR}/n_features_vs_samples.html',
    auto_open=True,
)
# %%
