"""Five-seed extension of the non-v6 final20 seed-bag experiment."""
import argparse
from pathlib import Path

from . import final20_seedbag as base


def main(resume=False):
    base.OUT = Path('runs/final20_seedbag5_v15')
    base.TREE_SEEDS = (0, 1, 2, 3, 4)
    base.run(resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.resume)
