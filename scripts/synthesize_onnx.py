#!/usr/bin/env python3
"""Synthesize a DiffSinger acoustic ONNX package with the local NSF-HiFiGAN vocoder."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--package-dir", required=True, type=pathlib.Path)
    parser.add_argument("--diffsinger-dir", required=True, type=pathlib.Path)
    parser.add_argument("--speaker", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--stream-dir", type=pathlib.Path)
    return parser.parse_args()


def curve_values(segment: dict, key: str, timestep_key: str, frame_count: int, timestep: float) -> np.ndarray:
    value = segment.get(key)
    if value is None:
        raise ValueError(f"{key} is required by the selected acoustic model")
    if isinstance(value, (int, float)):
        return np.full(frame_count, float(value), dtype=np.float32)
    points = np.asarray([float(item) for item in str(value).split()], dtype=np.float32)
    if not len(points):
        raise ValueError(f"{key} must contain at least one value")
    try:
        source_timestep = float(segment[timestep_key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{timestep_key} is required with {key}") from error
    if source_timestep <= 0:
        raise ValueError(f"{timestep_key} must be positive")
    target_times = np.arange(frame_count, dtype=np.float32) * timestep
    source_times = np.arange(len(points), dtype=np.float32) * source_timestep
    return np.interp(target_times, source_times, points).astype(np.float32)


def save_wav_atomically(waveform: np.ndarray, path: pathlib.Path, sample_rate: int, save_wav) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    save_wav(waveform.copy(), temporary_path, sample_rate)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    model_path = args.model.resolve()
    package_dir = args.package_dir.resolve()
    diffsinger_dir = args.diffsinger_dir.resolve()
    if not model_path.is_file() or model_path.parent != package_dir:
        raise SystemExit(f"Acoustic ONNX model is not in the selected package: {model_path}")

    config_path = package_dir / "dsconfig.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    phonemes_path = (package_dir / config["phonemes"]).resolve()
    languages_path = (package_dir / config["languages"]).resolve()
    if package_dir not in phonemes_path.parents or not phonemes_path.is_file():
        raise SystemExit("The acoustic ONNX package is missing its phoneme dictionary")
    if package_dir not in languages_path.parents or not languages_path.is_file():
        raise SystemExit("The acoustic ONNX package is missing its language dictionary")
    phoneme_ids = json.loads(phonemes_path.read_text(encoding="utf-8-sig"))
    language_ids = json.loads(languages_path.read_text(encoding="utf-8-sig"))
    try:
        segments = json.loads(args.input.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"JSON parse error: {error}") from error
    if isinstance(segments, dict):
        segments = segments.get("segments", [segments])
    if not isinstance(segments, list) or not segments:
        raise SystemExit(".ds data must contain at least one segment")

    sample_rate = int(config["sample_rate"])
    hop_size = int(config["hop_size"])
    timestep = hop_size / sample_rate
    expected_vocoder = {
        "sample_rate": 44_100,
        "hop_size": 512,
        "num_mel_bins": 128,
        "mel_base": "e",
    }
    incompatible = [
        key for key, value in expected_vocoder.items()
        if config.get(key) != value
    ]
    if incompatible:
        raise SystemExit(
            "The local PC-NSF-HiFiGAN requires matching ONNX mel settings; "
            f"incompatible: {', '.join(incompatible)}"
        )

    sys.path.insert(0, str(diffsinger_dir))
    os.chdir(diffsinger_dir)
    from utils.hparams import hparams, set_hparams

    set_hparams(config="configs/acoustic.yaml", print_hparams=False)
    from inference.ds_acoustic import DiffSingerAcousticInfer
    from utils.infer_utils import cross_fade, save_wav

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_nodes = {item.name: item for item in session.get_inputs()}
    missing_inputs = {"tokens", "durations", "f0"} - input_nodes.keys()
    if missing_inputs:
        raise ValueError(f"Acoustic ONNX is missing required inputs: {', '.join(sorted(missing_inputs))}")
    dtype_by_onnx_type = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    vocoder = DiffSingerAcousticInfer(load_model=False, load_vocoder=True)

    stream_dir = args.stream_dir
    if stream_dir:
        stream_dir.mkdir(parents=True, exist_ok=True)
    result = np.zeros(0, dtype=np.float32)
    current_length = 0
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"DS segment {index} must be an object")
        phones = str(segment.get("ph_seq", "")).split()
        try:
            phone_ids = [
                int(phoneme_ids.get(f"{segment.get('lang')}/{phone}", phoneme_ids.get(phone)))
                for phone in phones
            ]
            phone_durations = np.asarray(
                [float(value) for value in str(segment["ph_dur"]).split()],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid phonemes or durations in segment {index}: {error}") from error
        if not phones or len(phones) != len(phone_durations):
            raise ValueError(f"ph_seq and ph_dur lengths must match in segment {index}")
        if np.any(~np.isfinite(phone_durations)) or np.any(phone_durations < 0):
            raise ValueError(f"ph_dur must contain finite non-negative values in segment {index}")
        cumulative = np.rint(np.cumsum(phone_durations) / timestep).astype(np.int64)
        durations = np.diff(np.concatenate(([0], cumulative)))
        frame_count = int(durations.sum())
        if frame_count <= 0:
            raise ValueError(f"segment {index} has no positive-duration frames")

        f0 = curve_values(segment, "f0_seq", "f0_timestep", frame_count, timestep)
        language = segment.get("lang")
        feed = {
            "tokens": np.asarray([phone_ids], dtype=np.int64),
            "durations": durations[None, :],
            "f0": f0[None, :],
        }
        if "steps" in input_nodes:
            feed["steps"] = np.asarray(args.steps, dtype=np.int64)
        for name, node in input_nodes.items():
            if name in feed:
                continue
            if name == "depth":
                feed[name] = np.asarray(float(config.get("max_depth", 1.0)), dtype=np.float32)
            elif name == "languages":
                feed[name] = np.asarray([[
                    int(language_ids.get(f"{language}/{phone}", 0))
                    for phone in phones
                ]], dtype=np.int64)
            elif name == "gender":
                value = segment.get("gender", 0.0)
                feed[name] = (
                    np.full((1, frame_count), float(value), dtype=np.float32)
                    if isinstance(value, (int, float))
                    else curve_values(segment, "gender", "gender_timestep", frame_count, timestep)[None, :]
                )
            elif name == "velocity":
                if segment.get("velocity") is None:
                    feed[name] = np.ones((1, frame_count), dtype=np.float32)
                else:
                    feed[name] = curve_values(segment, "velocity", "velocity_timestep", frame_count, timestep)[None, :]
            elif name in {"energy", "breathiness", "voicing", "tension"}:
                feed[name] = curve_values(
                    segment, name, f"{name}_timestep", frame_count, timestep
                )[None, :]
            elif name == "spk_embed":
                speakers = config.get("speakers") or []
                selected_speaker = args.speaker or (speakers[0] if len(speakers) == 1 else "")
                if selected_speaker not in speakers:
                    raise ValueError("Select a speaker exported with this acoustic ONNX model")
                embedding_path = (package_dir / f"{selected_speaker}.emb").resolve()
                if package_dir not in embedding_path.parents or not embedding_path.is_file():
                    raise ValueError(f"Speaker embedding is missing: {selected_speaker}")
                embedding = np.fromfile(embedding_path, dtype=np.float32)
                hidden_size = int(config["hidden_size"])
                if embedding.size != hidden_size:
                    raise ValueError(f"Invalid speaker embedding size for {selected_speaker}")
                feed[name] = np.broadcast_to(
                    embedding.reshape(1, 1, hidden_size),
                    (1, frame_count, hidden_size),
                ).copy()
            else:
                raise ValueError(f"Unsupported acoustic ONNX input: {name}")
            dtype = dtype_by_onnx_type.get(node.type)
            if dtype is None:
                raise ValueError(f"Unsupported ONNX input type for {name}: {node.type}")
            feed[name] = np.asarray(feed[name], dtype=dtype)

        mel = session.run(None, feed)[0]
        if mel.ndim != 3 or mel.shape[1] != frame_count:
            raise ValueError(f"Unexpected mel shape from acoustic ONNX: {mel.shape}")
        waveform = vocoder.run_vocoder(
            torch.from_numpy(mel),
            f0=torch.from_numpy(f0[None, :]),
        )[0].cpu().numpy()
        if stream_dir:
            segment_path = stream_dir / f"segment-{index:06d}.wav"
            save_wav(waveform.copy(), segment_path, sample_rate)
        silent_length = round(float(segment.get("offset", 0)) * sample_rate) - current_length
        if silent_length >= 0:
            result = np.append(result, np.zeros(silent_length, dtype=np.float32))
            result = np.append(result, waveform)
        else:
            result = cross_fade(result, waveform, current_length + silent_length)
        current_length += silent_length + waveform.shape[0]
        if stream_dir:
            save_wav_atomically(
                result,
                stream_dir / "partial.wav",
                int(hparams["audio_sample_rate"]),
                save_wav,
            )
            print(json.dumps({
                "type": "segment",
                "index": index,
                "total": len(segments),
                "label": str(segment.get("text") or segment.get("ph_seq", ""))[:100],
                "duration": waveform.shape[0] / sample_rate,
                "offset": float(segment.get("offset", 0)),
            }, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_wav_atomically(result, args.output, int(hparams["audio_sample_rate"]), save_wav)
    print(args.output)


if __name__ == "__main__":
    main()
