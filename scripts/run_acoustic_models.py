#!/usr/bin/env python3
"""Generate and optionally train lightweight DiffSinger acoustic model variants."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CANDIDATES = (
    {"name": "tiny", "hidden_size": 32},
    {"name": "small", "hidden_size": 64},
    {"name": "medium", "hidden_size": 96},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffsinger-dir", type=Path,
        default=Path(os.environ.get("DIFFSINGER_DIR", ROOT_DIR / "third_party/DiffSinger")),
    )
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--binary-data-dir", default="data/acoustic_cpu/binary")
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--spk-id", type=int, default=0)
    parser.add_argument(
        "--test-prefix", action="append", required=True,
        help="Held-out utterance prefix; repeat for each validation utterance",
    )
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--val-check-interval", type=int, default=100)
    parser.add_argument("--train", action="store_true", help="Train each generated model after writing configs")
    return parser.parse_args()


def make_config(args: argparse.Namespace, template: dict, candidate: dict) -> dict:
    config = yaml.safe_load(yaml.safe_dump(template))
    config["datasets"] = [{
        "raw_data_dir": str(Path(args.raw_data_dir).resolve()),
        "speaker": args.speaker,
        "spk_id": args.spk_id,
        "language": args.language,
        "test_prefixes": args.test_prefix,
    }]
    config["dictionaries"] = {args.language: str(Path(args.dictionary).resolve())}
    config["binary_data_dir"] = args.binary_data_dir
    config["binarization_args"]["num_workers"] = 0
    config["binarization_args"]["allow_unseen_phonemes"] = True

    hidden_size = candidate["hidden_size"]
    config["hidden_size"] = hidden_size
    config["enc_layers"] = 1
    config["use_shallow_diffusion"] = False
    config["use_fused_kernels"] = False
    config["backbone_args"].update({
        "num_channels": hidden_size,
        "num_layers": 1,
        "kernel_size": 31,
        "use_conditioner_cache": False,
    })

    # Training and validation do not require a vocoder; synthesis does.
    config["val_with_vocoder"] = False
    config["max_batch_frames"] = 12000
    config["max_batch_size"] = 4
    config["max_val_batch_frames"] = 12000
    config["max_val_batch_size"] = 1
    config["max_updates"] = args.max_updates
    config["val_check_interval"] = args.val_check_interval
    config["pl_trainer_accelerator"] = "cpu"
    config["pl_trainer_devices"] = 1
    config["pl_trainer_precision"] = "32-true"
    config["pl_trainer_strategy"] = {
        "name": "auto",
        "process_group_backend": "gloo",
        "find_unused_parameters": False,
    }
    config["nccl_p2p"] = False
    return config


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    if args.max_updates <= 0 or args.val_check_interval <= 0:
        raise SystemExit("--max-updates and --val-check-interval must be positive")
    if args.max_updates % args.val_check_interval != 0:
        raise SystemExit("--max-updates must be divisible by --val-check-interval")

    diffsinger_dir = args.diffsinger_dir.resolve()
    template_path = diffsinger_dir / "configs/templates/config_acoustic.yaml"
    with template_path.open(encoding="utf-8") as file:
        template = yaml.safe_load(file)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    configs = {}
    for candidate in CANDIDATES:
        config_path = diffsinger_dir / "configs" / f"acoustic_cpu_{candidate['name']}_{run_id}.yaml"
        config = make_config(args, template, candidate)
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        configs[candidate["name"]] = config_path
        print(f"{candidate['name']}: {config_path}")

    if not args.train:
        print("Configs generated. Pass --train to binarize the data and train all variants.")
        return

    venv_dir = Path(os.environ.get("VENV_DIR", ROOT_DIR / ".venv-cpu312")).resolve()
    python = str(venv_dir / "bin/python")
    if not Path(python).is_file():
        raise SystemExit("CPU environment not found. Run scripts/setup_cpu.sh first.")

    run([python, "scripts/binarize.py", "--config", str(next(iter(configs.values())))], diffsinger_dir)
    for name, config_path in configs.items():
        run([
            python, "scripts/train.py", "--config", str(config_path),
            "--exp_name", f"acoustic_cpu_{run_id}_{name}",
        ], diffsinger_dir)


if __name__ == "__main__":
    main()
