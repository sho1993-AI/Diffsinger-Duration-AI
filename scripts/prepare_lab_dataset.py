#!/usr/bin/env python3
"""Convert aligned WAV/LAB pairs in a ZIP into DiffSinger raw-data format."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import wave
import zipfile


TICKS_PER_SECOND = 10_000_000
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
VOWELS = {"a", "i", "u", "e", "o"}
SYLLABLE_BOUNDARIES = {"AP", "SP", "pau", "sil", "N", "q", "cl"}


def make_ph_num(phones: list[str]) -> list[int]:
    groups = []
    current_size = 0
    for phone in phones:
        if phone in SYLLABLE_BOUNDARIES:
            if current_size:
                groups.append(current_size)
                current_size = 0
            groups.append(1)
            continue
        current_size += 1
        if phone in VOWELS:
            groups.append(current_size)
            current_size = 0
    if current_size:
        groups.append(current_size)
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument(
        "--dictionary-out", type=pathlib.Path,
        default=pathlib.Path("data/duration_cpu/phonemes.txt"),
    )
    parser.add_argument(
        "--phoneme-inventory-archive", type=pathlib.Path,
        default=ROOT_DIR / "data/sho2.zip",
        help="SHO ZIP containing sho2/phonemes.json; adds symbols even when not in labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        members = {member.filename: member for member in archive.infolist() if not member.is_dir()}
        wav_members = sorted(name for name in members if name.lower().endswith(".wav"))
        if not wav_members:
            raise SystemExit(f"No WAV files found in {args.archive}")

        rows = []
        all_phones = {"AP", "SP"}
        for wav_member in wav_members:
            lab_member = wav_member[:-4] + ".lab"
            if lab_member not in members:
                raise SystemExit(f"Missing paired LAB file: {lab_member}")

            stem = pathlib.PurePosixPath(wav_member).stem
            wav_bytes = archive.read(wav_member)
            with wave.open(archive.open(wav_member), "rb") as wav_file:
                wav_duration = wav_file.getnframes() / wav_file.getframerate()

            phone_seq = []
            phone_durations = []
            previous_end = 0
            lab_text = archive.read(lab_member).decode("utf-8-sig")
            for line_number, line in enumerate(lab_text.splitlines(), start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 3:
                    raise SystemExit(f"{lab_member}:{line_number}: expected start end phoneme")
                try:
                    start, end = int(fields[0]), int(fields[1])
                except ValueError as error:
                    raise SystemExit(f"{lab_member}:{line_number}: invalid timestamp") from error
                if start < previous_end or end < start:
                    raise SystemExit(f"{lab_member}:{line_number}: timestamps overlap or go backwards")
                phone = fields[2]
                phone_seq.append(phone)
                phone_durations.append((end - start) / TICKS_PER_SECOND)
                all_phones.add(phone)
                previous_end = end

            label_end = previous_end / TICKS_PER_SECOND
            if not phone_seq:
                raise SystemExit(f"No phoneme labels found in {lab_member}")
            if label_end > wav_duration + 0.5:
                raise SystemExit(
                    f"{lab_member}: labels end at {label_end:.3f}s, "
                    f"but WAV is only {wav_duration:.3f}s"
                )

            (wav_dir / f"{stem}.wav").write_bytes(wav_bytes)
            rows.append({
                "name": stem,
                "ph_seq": " ".join(phone_seq),
                "ph_dur": " ".join(f"{duration:.8f}" for duration in phone_durations),
                "ph_num": " ".join(str(count) for count in make_ph_num(phone_seq)),
            })
            print(f"{stem}: {len(phone_seq)} phones, {label_end:.2f}s aligned")

    inventory_archive = args.phoneme_inventory_archive.resolve()
    if inventory_archive.is_file():
        with zipfile.ZipFile(inventory_archive) as archive:
            try:
                inventory = json.loads(archive.read("sho2/phonemes.json"))
            except KeyError as error:
                raise SystemExit(f"Missing sho2/phonemes.json in {inventory_archive}") from error
        all_phones.update(inventory)
        print(f"Added {len(inventory)} phoneme inventory entries from {inventory_archive}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "transcriptions.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("name", "ph_seq", "ph_dur", "ph_num"))
        writer.writeheader()
        writer.writerows(rows)

    dictionary_path = args.dictionary_out.resolve()
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    with dictionary_path.open("w", encoding="utf-8", newline="") as dictionary_file:
        for phone in sorted(all_phones):
            dictionary_file.write(f"{phone}\t{phone}\n")

    print(f"Raw dataset: {output_dir}")
    print(f"Phoneme dictionary: {dictionary_path}")


if __name__ == "__main__":
    main()