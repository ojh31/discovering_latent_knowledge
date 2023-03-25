import sys
from dlk.utils import get_parser
from dlk.generate import run_gen

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    run_gen(args)