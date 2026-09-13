"""Controlled 50-fold variant with a small frozen iteration adjustment.

It reuses final_refit50_fixed and changes only the LightGBM/CatBoost iteration
count by a fixed scale.  No parameters or features are re-selected.
"""
import argparse
from pathlib import Path

from . import final_refit50_fixed as base


def main(resume=False, output=None, iteration_scale=1.03):
    if output is not None:
        base.OUT = Path(output)
    original_loader = base.load_recipe

    def scaled_loader():
        configs, iterations = original_loader()
        scaled = {
            family: max(1, int(round(value * iteration_scale)))
            for family, value in iterations.items()
        }
        return configs, scaled

    base.load_recipe = scaled_loader
    base.run(resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', default='runs/final50_scaled_v13')
    parser.add_argument('--iteration-scale', type=float, default=1.03)
    args = parser.parse_args()
    main(args.resume, args.output, args.iteration_scale)
