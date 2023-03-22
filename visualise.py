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
df['model_name'] = df.full_name.str.split('__').str[0]
df['dataset_name'] = df.full_name.str.split('__').str[1]
df['key'] = df.full_name.str.split('__').str[2]
df['split'] = df.key.str.split('_').str[1]
df['reg'] = df.key.str.split('_').str[0]
df
# %%
# Train vs. test
fig = px.line(
    df, x='split', y='accuracy', color='dataset_name', 
    facet_col='reg', facet_row='model_name'
)
fig.show()

# %%
