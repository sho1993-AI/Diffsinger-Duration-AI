#!/usr/bin/env python3
"""Train and compare the official DiffSinger duration predictor variants on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


DIFFSINGER_COMMIT = "cb7869b8733a3882679c328994ee162845092cff"
CANDIDATES = (
    {
        "name": "attn_tiny", "arch": "attn", "hidden_size": 64, "enc_layers": 2,
        "dur_hidden_size": 64, "dur_layers": 1, "dur_num_blocks": 1, "dur_num_heads": 2,
    },
    {
        "name": "fs2_tiny", "arch": "fs2", "hidden_size": 64, "enc_layers": 2,
        "dur_hidden_size": 64, "dur_layers": 1, "dur_num_blocks": 1, "dur_num_heads": 2,
    },
    {
        "name": "resnet_tiny", "arch": "resnet", "hidden_size": 64, "enc_layers": 2,
        "dur_hidden_size": 64, "dur_layers": 1, "dur_num_blocks": 1, "dur_num_heads": 2,
    },
    {
        "name": "fs2_small", "arch": "fs2", "hidden_size": 96, "enc_layers": 2,
        "dur_hidden_size": 96, "dur_layers": 2, "dur_num_blocks": 1, "dur_num_heads": 2,
    },
    {
        "name": "attn_small", "arch": "attn", "hidden_size": 96, "enc_layers": 2,
        "dur_hidden_size": 96, "dur_layers": 2, "dur_num_blocks": 2, "dur_num_heads": 4,
    },
    {
        "name": "attn_small_60", "arch": "attn", "hidden_size": 96, "enc_layers": 2,
        "dur_hidden_size": 96, "dur_layers": 2, "dur_num_blocks": 2, "dur_num_heads": 4,
        "update_multiplier": 2,
    },
)
ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffsinger-dir", type=Path,
        default=Path(os.environ.get("DIFFSINGER_DIR", ROOT_DIR / "third_party/DiffSinger")),
    )
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--binary-data-dir", default="data/duration_cpu/binary_allphones")
    parser.add_argument("--dictionary", required=True, help="DiffSinger phoneme dictionary file")
    parser.add_argument("--language", required=True, help="Dictionary language key, e.g. en or ja")
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--spk-id", type=int, default=0)
    parser.add_argument(
        "--test-prefix", action="append", required=True,
        help="Held-out utterance prefix; repeat for each validation utterance",
    )
    parser.add_argument("--max-updates", type=int, default=30)
    parser.add_argument("--val-check-interval", type=int, default=10)
    parser.add_argument(
        "--accuracy-tolerance", type=float, default=0.02,
        help="Choose the smallest/fastest candidate within this absolute accuracy of the best",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Write configs without preprocessing or training")
    return parser.parse_args()


def verify_checkout(diffsinger_dir: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(diffsinger_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip() != DIFFSINGER_COMMIT:
        raise SystemExit(
            f"Expected DiffSinger commit {DIFFSINGER_COMMIT}, found {result.stdout.strip()}"
        )


def make_config(args: argparse.Namespace, template: dict, candidate: dict) -> dict:
    config = json.loads(json.dumps(template))
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
    config["predict_dur"] = True
    config["predict_pitch"] = False
    config["predict_energy"] = False
    config["predict_breathiness"] = False
    config["predict_voicing"] = False
    config["predict_tension"] = False
    config["use_fused_kernels"] = False
    config["ds_workers"] = 1
    config["hidden_size"] = candidate["hidden_size"]
    config["enc_layers"] = candidate["enc_layers"]
    config["dur_prediction_args"].update({
        "arch": candidate["arch"],
        "hidden_size": candidate["dur_hidden_size"],
        "dropout": 0.1,
        "num_layers": candidate["dur_layers"],
        "dur_num_blocks": candidate["dur_num_blocks"],
        "dur_num_heads": candidate["dur_num_heads"],
        "dur_ffn_mult": 2,
    })
    config["max_updates"] = args.max_updates * candidate.get("update_multiplier", 1)
    config["val_check_interval"] = args.val_check_interval
    config["max_batch_frames"] = 20000
    config["max_batch_size"] = 16
    config["max_val_batch_frames"] = 20000
    config["max_val_batch_size"] = 1
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


def latest_scalar(experiment_dir: Path, tag: str) -> tuple[int, float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values: list[tuple[int, float]] = []
    event_files = sorted(experiment_dir.rglob("events.out.tfevents.*"))
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags().get("scalars", []):
            values.extend((item.step, item.value) for item in accumulator.Scalars(tag))
    if not values:
        raise RuntimeError(f"TensorBoard scalar {tag!r} not found under {experiment_dir}")
    return max(values, key=lambda item: item[0])


def checkpoint_for(experiment_dir: Path) -> tuple[int, Path]:
    checkpoints = []
    for path in experiment_dir.glob("model_ckpt_steps_*.ckpt"):
        match = re.fullmatch(r"model_ckpt_steps_(\d+)\.ckpt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise RuntimeError(f"No model checkpoint found under {experiment_dir}")
    return max(checkpoints, key=lambda item: item[0])


def main() -> None:
    args = parse_args()
    if args.max_updates <= 0 or args.val_check_interval <= 0:
        raise SystemExit("--max-updates and --val-check-interval must be positive")
    if args.max_updates % args.val_check_interval != 0:
        raise SystemExit("--max-updates must be divisible by --val-check-interval")
    if args.accuracy_tolerance < 0:
        raise SystemExit("--accuracy-tolerance must be non-negative")

    diffsinger_dir = args.diffsinger_dir.resolve()
    verify_checkout(diffsinger_dir)
    template_path = diffsinger_dir / "configs/templates/config_duration.yaml"
    with template_path.open(encoding="utf-8") as file:
        template = yaml.safe_load(file)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    configs: dict[str, Path] = {}
    with (Path(args.raw_data_dir) / "transcriptions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    has_ap = any("AP" in row["ph_seq"].split() for row in rows)
    with Path(args.dictionary).open(encoding="utf-8") as file:
        dictionary_phones = {line.split("\t", maxsplit=1)[0] for line in file if line.strip()}
    observed_phones = {
        phone for row in rows for phone in row["ph_seq"].split()
    }
    untrained_phones = sorted(dictionary_phones - observed_phones)
    if not has_ap and "SP" in observed_phones and "AP" in untrained_phones:
        untrained_phones.remove("AP")
    validation_names = [
        row["name"] for row in rows
        if any(row["name"].startswith(prefix) for prefix in args.test_prefix)
    ]
    if not validation_names:
        raise SystemExit("No validation utterances matched --test-prefix")

    for candidate in CANDIDATES:
        config_path = diffsinger_dir / "configs" / f"duration_cpu_{candidate['name']}_{run_id}.yaml"
        config = make_config(args, template, candidate)
        if not has_ap:
            config["merged_phoneme_groups"] = [*config.get("merged_phoneme_groups", []), ["AP", "SP"]]
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        configs[candidate["name"]] = config_path

    if args.prepare_only:
        for name, config_path in configs.items():
            print(f"{name}: {config_path}")
        return

    venv_dir = Path(os.environ.get("VENV_DIR", ROOT_DIR / ".venv-cpu312")).resolve()
    python = str(venv_dir / "bin/python")
    if not Path(python).is_file():
        raise SystemExit("CPU environment not found. Run scripts/setup_cpu.sh first.")

    run([python, "scripts/binarize.py", "--config", str(next(iter(configs.values())))], diffsinger_dir)

    experiments: dict[str, str] = {}
    elapsed_by_candidate = {}
    for candidate in CANDIDATES:
        name = candidate["name"]
        config_path = configs[name]
        experiment = f"duration_cpu_{run_id}_{name}"
        start_time = time.perf_counter()
        run([
            python, "scripts/train.py", "--config", str(config_path),
            "--exp_name", experiment,
        ], diffsinger_dir)
        elapsed_by_candidate[name] = time.perf_counter() - start_time
        experiments[name] = experiment

    results = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        experiment = experiments[name]
        experiment_dir = diffsinger_dir / "checkpoints" / experiment
        acc_step, ph_dur_acc = latest_scalar(experiment_dir, "metrics/ph_dur_acc")
        corr_step, rhythm_corr = latest_scalar(experiment_dir, "metrics/rhythm_corr")
        loss_step, dur_loss = latest_scalar(experiment_dir, "validation/dur_loss")
        checkpoint_step, checkpoint = checkpoint_for(experiment_dir)
        checkpoint_bytes = checkpoint.stat().st_size
        train_seconds = elapsed_by_candidate[name]
        results.append({
            "name": name,
            "architecture": candidate["arch"],
            "parameters": candidate,
            "experiment": experiment,
            "ph_dur_acc": ph_dur_acc,
            "rhythm_corr": rhythm_corr,
            "dur_loss": dur_loss,
            "train_seconds": train_seconds,
            "seconds_per_update": train_seconds / max(checkpoint_step, 1),
            "checkpoint_bytes": checkpoint_bytes,
            "metric_steps": {"ph_dur_acc": acc_step, "rhythm_corr": corr_step, "dur_loss": loss_step},
            "checkpoint_step": checkpoint_step,
            "checkpoint": str(checkpoint),
        })

    results.sort(key=lambda result: (-result["ph_dur_acc"], -result["rhythm_corr"], result["dur_loss"]))
    best_accuracy = results[0]["ph_dur_acc"]
    efficient_candidates = [
        result for result in results
        if result["ph_dur_acc"] >= best_accuracy - args.accuracy_tolerance
    ]
    selected = min(efficient_candidates, key=lambda result: (
        result["checkpoint_bytes"] * result["seconds_per_update"],
        result["checkpoint_bytes"],
        result["seconds_per_update"],
    ))
    report = {
        "selection_rule": (
            "among candidates within accuracy_tolerance of best validation ph_dur_acc, "
            "minimize checkpoint_bytes * seconds_per_update"
        ),
        "accuracy_tolerance": args.accuracy_tolerance,
        "training_utterances": len(rows) - len(validation_names),
        "validation_utterances": validation_names,
        "untrained_phonemes": untrained_phones,
        "best_accuracy_candidate": results[0],
        "best": selected,
        "candidates": results,
    }
    report_path = ROOT_DIR / f"duration_sweep_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nBest Duration model: {selected['experiment']}")
    print(f"Checkpoint: {selected['checkpoint']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()