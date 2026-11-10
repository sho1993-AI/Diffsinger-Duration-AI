#!/usr/bin/env python3
"""Run the official DiffSinger Duration predictor on a .ds file."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from duration_postprocess import calibrate_duration_file


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DIFFSINGER_DIR = ROOT_DIR / "third_party/DiffSinger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--diffsinger-dir", type=pathlib.Path, default=DEFAULT_DIFFSINGER_DIR)
    parser.add_argument("--title", default="duration_prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diffsinger_dir = args.diffsinger_dir.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input .ds file not found: {input_path}")
    experiment_dir = diffsinger_dir / "checkpoints" / args.experiment
    if not (experiment_dir / "config.yaml").is_file():
        raise SystemExit(f"DiffSinger experiment not found: {args.experiment}")

    sys.path.insert(0, str(diffsinger_dir))
    os.chdir(diffsinger_dir)
    sys.argv = ["duration-inference", "--exp_name", args.experiment, "--infer"]
    from utils.hparams import set_hparams

    set_hparams()

    from inference.ds_variance import DiffSingerVarianceInfer

    params = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if isinstance(params, dict):
        params = [params]
    if not isinstance(params, list) or not params:
        raise SystemExit("Input .ds must contain at least one segment")

    infer = DiffSingerVarianceInfer(
        ckpt_steps=args.checkpoint,
        predictions={"dur"},
    )
    infer.run_inference(
        params,
        out_dir=output_dir,
        title=args.title,
        num_runs=1,
        seed=-1,
    )
    output_path = output_dir / f"{args.title}.ds"
    adjusted = calibrate_duration_file(output_path)
    print(f"Calibrated {adjusted} leading phonemes to training means (max {0.3:.1f}s).")
    print(output_path)


if __name__ == "__main__":
    main()
