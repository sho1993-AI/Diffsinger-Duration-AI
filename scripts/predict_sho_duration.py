#!/usr/bin/env python3
"""Predict Japanese phoneme durations with the pretrained SHO ONNX model."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import zipfile

import numpy as np
import onnxruntime as ort

from duration_postprocess import calibrate_duration_file


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT_DIR / "data/sho2.zip"
SAMPLE_RATE = 44_100
HOP_SIZE = 512
NOTE_PATTERN = re.compile(r"^([A-G])([#b]?)(-?\d+)$")
NOTE_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(note: str) -> int | None:
    if note.lower() == "rest":
        return None
    match = NOTE_PATTERN.fullmatch(note)
    if not match:
        raise ValueError(f"Unsupported note name: {note}")
    letter, accidental, octave_text = match.groups()
    accidental_value = 1 if accidental == "#" else -1 if accidental == "b" else 0
    return (int(octave_text) + 1) * 12 + NOTE_SEMITONES[letter] + accidental_value


def note_groups(slurs: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for index, is_slur in enumerate(slurs):
        if not is_slur or not groups:
            groups.append([])
        groups[-1].append(index)
    return groups


def expand_rest_notes(midis: list[int | None]) -> list[int]:
    sounding = [index for index, midi in enumerate(midis) if midi is not None]
    if not sounding:
        return [60] * len(midis)
    return [
        int(midi) if midi is not None else int(midis[min(sounding, key=lambda index: abs(index - i))])
        for i, midi in enumerate(midis)
    ]


def phoneme_midi_for_word(
    phone_durations: list[float],
    note_indices: list[int],
    note_durations: list[float],
    note_midis: list[int],
) -> list[int]:
    phone_weights = [max(value, 0) for value in phone_durations]
    phone_total = sum(phone_weights)
    if phone_total <= 0:
        phone_weights = [1.0] * len(phone_weights)
        phone_total = len(phone_weights)

    note_weights = [max(note_durations[index], 0) for index in note_indices]
    note_total = sum(note_weights)
    if note_total <= 0:
        note_weights = [1.0] * len(note_indices)
        note_total = len(note_indices)

    result = []
    phone_elapsed = 0.0
    for weight in phone_weights:
        midpoint = (phone_elapsed + weight / 2) / phone_total
        note_elapsed = 0.0
        selected = note_indices[-1]
        for note_index, note_weight in zip(note_indices, note_weights):
            note_elapsed += note_weight / note_total
            if midpoint <= note_elapsed:
                selected = note_index
                break
        result.append(note_midis[selected])
        phone_elapsed += weight
    return result


def make_inputs(segment: dict, phoneme_ids: dict[str, int]) -> dict[str, np.ndarray]:
    phones = str(segment.get("ph_seq", "")).split()
    word_div = [int(value) for value in str(segment.get("ph_num", "")).split()]
    notes = str(segment.get("note_seq", "")).split()
    note_durations = [float(value) for value in str(segment.get("note_dur", "")).split()]
    note_slurs = [int(value) for value in str(segment.get("note_slur", "")).split()]
    source_durations = [float(value) for value in str(segment.get("ph_dur", "")).split()]

    if not phones or not word_div or sum(word_div) != len(phones):
        raise ValueError("ph_seq and ph_num must describe the same phonemes")
    if len(notes) != len(note_durations) or len(notes) != len(note_slurs):
        raise ValueError("note_seq, note_dur, and note_slur lengths must match")
    groups = note_groups(note_slurs)
    if len(groups) != len(word_div):
        raise ValueError("note_slur groups must match the words described by ph_num")
    if source_durations and len(source_durations) != len(phones):
        raise ValueError("ph_dur length must match ph_seq when provided")

    unknown = sorted(set(phones) - phoneme_ids.keys())
    if unknown:
        raise ValueError(f"SHO model does not contain these phonemes: {' '.join(unknown)}")

    phone_midis: list[int] = []
    word_durations = []
    phone_offset = 0
    note_midis = expand_rest_notes([note_to_midi(note) for note in notes])
    for word_index, (phone_count, group) in enumerate(zip(word_div, groups)):
        group_duration = sum(note_durations[index] for index in group)
        word_durations.append(max(0, math.floor(group_duration * SAMPLE_RATE / HOP_SIZE + 0.5)))
        if source_durations:
            weights = source_durations[phone_offset:phone_offset + phone_count]
        else:
            weights = [1.0] * phone_count
        phone_midis.extend(phoneme_midi_for_word(weights, group, note_durations, note_midis))
        phone_offset += phone_count

    return {
        "tokens": np.asarray([[phoneme_ids[phone] for phone in phones]], dtype=np.int64),
        "word_div": np.asarray([word_div], dtype=np.int64),
        "word_dur": np.asarray([word_durations], dtype=np.int64),
        "ph_midi": np.asarray([phone_midis], dtype=np.int64),
    }


def predict_segments(segments: list[dict], archive_path: pathlib.Path = DEFAULT_ARCHIVE) -> list[dict]:
    with zipfile.ZipFile(archive_path) as archive:
        phoneme_ids = json.loads(archive.read("sho2/phonemes.json"))
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, min(4, ort.get_device() == "CPU" and 2 or 1))
        linguistic = ort.InferenceSession(
            archive.read("sho2/dsdur/linguistic.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        duration_model = ort.InferenceSession(
            archive.read("sho2/dsdur/dur.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    predicted = []
    for segment in segments:
        inputs = make_inputs(segment, phoneme_ids)
        encoder_out, x_masks = linguistic.run(
            None,
            {key: inputs[key] for key in ("tokens", "word_div", "word_dur")},
        )
        frame_durations = duration_model.run(
            None,
            {
                "encoder_out": encoder_out,
                "x_masks": x_masks,
                "ph_midi": inputs["ph_midi"],
            },
        )[0][0]
        result = dict(segment)
        result["ph_dur"] = " ".join(
            f"{max(0.0, float(value)) * HOP_SIZE / SAMPLE_RATE:.6f}"
            for value in frame_durations
        )
        result["ph_dur_model"] = "SHO pretrained ONNX"
        predicted.append(result)
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    segments = data if isinstance(data, list) else data.get("segments", [data])
    results = predict_segments(segments, args.archive.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    adjusted = calibrate_duration_file(args.output)
    print(f"Calibrated {adjusted} leading phonemes to training means (max {0.3:.1f}s).")
    print(args.output)


if __name__ == "__main__":
    main()
