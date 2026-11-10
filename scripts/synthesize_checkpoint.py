#!/usr/bin/env python3
"""Synthesize a DiffSinger checkpoint one phrase at a time."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--diffsinger-dir", required=True, type=pathlib.Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--speaker", default="")
    parser.add_argument("--stream-dir", required=True, type=pathlib.Path)
    return parser.parse_args()


def emit_segment(index: int, total: int, segment: dict, duration: float) -> None:
    print(json.dumps({
        "type": "segment",
        "index": index,
        "total": total,
        "label": str(segment.get("text") or segment.get("ph_seq", ""))[:100],
        "duration": duration,
        "offset": float(segment.get("offset", 0)),
    }, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    diffsinger_dir = args.diffsinger_dir.resolve()
    sys.path.insert(0, str(diffsinger_dir))
    sys.argv = [sys.argv[0], "--exp_name", args.experiment, "--infer"]

    from utils.hparams import hparams, set_hparams

    set_hparams()
    if "diff_speedup" not in hparams and "pndm_speedup" in hparams:
        hparams["diff_speedup"] = hparams["pndm_speedup"]
    if "T_start" not in hparams:
        hparams["T_start"] = 1 - hparams["K_step"] / hparams["timesteps"]
    if "T_start_infer" not in hparams:
        hparams["T_start_infer"] = 1 - hparams["K_step_infer"] / hparams["timesteps"]
    if "sampling_steps" not in hparams:
        if hparams["use_shallow_diffusion"]:
            hparams["sampling_steps"] = hparams["K_step_infer"] // hparams["diff_speedup"]
        else:
            hparams["sampling_steps"] = hparams["timesteps"] // hparams["diff_speedup"]
    if "time_scale_factor" not in hparams:
        hparams["time_scale_factor"] = hparams["timesteps"]

    from inference.ds_acoustic import DiffSingerAcousticInfer
    from utils.infer_utils import cross_fade, parse_commandline_spk_mix, save_wav

    params = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if isinstance(params, dict):
        params = params.get("segments", [params])
    if not isinstance(params, list) or not params:
        raise ValueError(".ds data must contain at least one segment")
    if args.speaker and hparams["use_spk_id"]:
        speaker_mix = parse_commandline_spk_mix(args.speaker)
        for param in params:
            param["spk_mix"] = speaker_mix

    args.stream_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.experiment} checkpoint {args.checkpoint}...", flush=True)
    infer = DiffSingerAcousticInfer(load_vocoder=True, ckpt_steps=args.checkpoint)
    result = np.zeros(0)
    current_length = 0
    sample_rate = int(hparams["audio_sample_rate"])

    for index, param in enumerate(params, start=1):
        if not isinstance(param, dict):
            raise ValueError(f"DS segment {index} must be an object")
        if "seed" in param:
            torch.manual_seed(param["seed"] & 0xFFFF_FFFF)
            torch.cuda.manual_seed_all(param["seed"] & 0xFFFF_FFFF)

        batch = infer.preprocess_input(param, idx=index - 1)
        mel = infer.forward_model(batch)
        waveform = infer.run_vocoder(mel, f0=batch["f0"])[0].cpu().numpy()
        save_wav(
            waveform.copy(),
            args.stream_dir / f"segment-{index:06d}.wav",
            sample_rate,
        )

        silent_length = round(float(param.get("offset", 0)) * sample_rate) - current_length
        if silent_length >= 0:
            result = np.append(result, np.zeros(silent_length))
            result = np.append(result, waveform)
        else:
            result = cross_fade(result, waveform, current_length + silent_length)
        current_length += silent_length + waveform.shape[0]

        temporary_path = args.stream_dir / "partial.wav.tmp"
        save_wav(result.copy(), temporary_path, sample_rate)
        temporary_path.replace(args.stream_dir / "partial.wav")
        emit_segment(index, len(params), param, waveform.shape[0] / sample_rate)

    temporary_path = args.output.with_name(args.output.name + ".tmp")
    save_wav(result.copy(), temporary_path, sample_rate)
    temporary_path.replace(args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
