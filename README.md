# Discovering Latent Knowledge Without Supervision

## Usage
Generation of contrast pair datasets with model hidden states and  labels is separated from the model fitting/evaluation scripts to allow for caching.

From the command line, use
```
wandb sweep evaluate_sweep.yaml 
```
for sweeps or 
```
python evaluate_runner.py --device cuda --model gpt2-l
```
for passing arguments in-line. Generation is similar to evaluation.


## Citation

Thank you to the authors of the original paper:

    @article{burns2022dl,
      title={Discovering Latent Knowledge in Language Models Without Supervision},
      author={Burns, Collin and Ye, Haotian and Klein, Dan and Steinhardt, Jacob},
      journal={ArXiV},
      year={2022}
    }

