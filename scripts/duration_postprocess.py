#!/usr/bin/env python3
"""Calibrate phrase-leading phoneme durations using labeled training means."""

from __future__ import annotations

import csv
import json
import pathlib
import statistics


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
MAX_LEADING_DURATION = 0.3
SHORTHAND_PAUSES = {"AP", "SP", "SIL", "PAU", "REST", "N", "Q", "CL"}
VOWELS = {"a", "i", "u", "e", "o"}


def load_duration_means() -> tuple[dict[str, float], float, float]:
    training_csv = ROOT_DIR / "data/duration_cpu/raw_mora/transcriptions.csv"
    if not training_csv.is_file():
        training_csv = ROOT_DIR / "data/duration_cpu/raw/transcriptions.csv"
    if not training_csv.is_file():
        return {}, 0.1, 0.1

    durations_by_phone: dict[str, list[float]] = {}
    vowel_durations = []
    consonant_durations = []
    with training_csv.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            phones = row["ph_seq"].split()
            durations = [float(value) for value in row["ph_dur"].split()]
            for phone, duration in zip(phones, durations):
                if duration <= 0 or phone.upper() in SHORTHAND_PAUSES:
                    continue
                durations_by_phone.setdefault(phone, []).append(duration)
                (vowel_durations if phone.lower() in VOWELS else consonant_durations).append(duration)

    means = {phone: statistics.fmean(values) for phone, values in durations_by_phone.items()}
    vowel_mean = statistics.fmean(vowel_durations) if vowel_durations else 0.1
    consonant_mean = statistics.fmean(consonant_durations) if consonant_durations else 0.1
    return means, vowel_mean, consonant_mean


def calibrate_leading_durations(
    segments: list[dict],
    means: dict[str, float],
    vowel_mean: float,
    consonant_mean: float,
    max_duration: float = MAX_LEADING_DURATION,
    mean_blend: float = 0.35,
) -> int:
    adjusted = 0
    for segment in segments:
        phones = str(segment.get("ph_seq", "")).split()
        durations = [float(value) for value in str(segment.get("ph_dur", "")).split()]
        if not phones or len(phones) != len(durations):
            continue

        leading_index = next(
            (index for index, phone in enumerate(phones) if phone.upper() not in SHORTHAND_PAUSES),
            None,
        )
        if leading_index is None:
            continue
        leading_phone = phones[leading_index]
        average = means.get(
            leading_phone,
            vowel_mean if leading_phone.lower() in VOWELS else consonant_mean,
        )
        current = durations[leading_index]
        target = min(max_duration, average + mean_blend * max(0.0, current - average))
        if current <= target:
            continue

        next_vowel = next(
            (index for index in range(leading_index + 1, len(phones))
             if phones[index].lower() in VOWELS),
            None,
        )
        recipients = [next_vowel] if next_vowel is not None else []
        if not recipients:
            next_phone = next(
                (index for index in range(leading_index + 1, len(phones))
                 if phones[index].upper() not in SHORTHAND_PAUSES),
                None,
            )
            recipients = [next_phone] if next_phone is not None else []
        if not recipients:
            continue
        excess = current - target
        weights = [max(0.0, durations[index]) for index in recipients]
        if sum(weights) <= 0:
            weights = [
                max(0.0, means.get(
                    phones[index],
                    vowel_mean if phones[index].lower() in VOWELS else consonant_mean,
                ))
                for index in recipients
            ]
        if sum(weights) <= 0:
            weights = [1.0] * len(recipients)
        weight_total = sum(weights)
        durations[leading_index] = target
        for index, weight in zip(recipients, weights):
            durations[index] += excess * weight / weight_total
        segment["ph_dur"] = " ".join(f"{duration:.6f}" for duration in durations)
        adjusted += 1
    return adjusted


def calibrate_duration_file(path: pathlib.Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        segments = data.get("segments", [data])
    else:
        segments = data
    means, vowel_mean, consonant_mean = load_duration_means()
    adjusted = calibrate_leading_durations(segments, means, vowel_mean, consonant_mean)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return adjusted
