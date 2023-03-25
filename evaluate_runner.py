import argparse
import sys
from typing import List
from dlk.evaluate import run_eval
from dlk.utils import get_parser

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
    parser.add_argument('--eval_path', type=str, default='results.json')
    parser.add_argument('--plot_dir', type=str, default='plots')
    parser.add_argument('--wandb_enabled', action='store_true')
    args = parser.parse_args(argv)
    return generation_args, args


if __name__ == '__main__':
    generation_args, args = parse_args(sys.argv[1:])
    run_eval(generation_args=generation_args, args=args)
