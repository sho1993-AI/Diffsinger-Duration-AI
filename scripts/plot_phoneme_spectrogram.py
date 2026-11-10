#!/usr/bin/env python3
"""Plot labeled WAV spectrograms and time-normalized mean phoneme spectra."""

from __future__ import annotations

import argparse
import io
import pathlib
import wave
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TICKS_PER_SECOND = 10_000_000
FFT_SIZE = 1024
AVERAGE_FRAMES = 96
MAX_FREQUENCY = 8_000
DYNAMIC_RANGE_DB = 80
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path, help="ZIP containing paired WAV and HTK LAB files")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT_DIR / "data/phoneme_spectrogram.png",
        help="Output PNG path (default: data/phoneme_spectrogram.png)",
    )
    return parser.parse_args()


def decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    elif sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = data[:, 0].astype(np.int32) | (data[:, 1].astype(np.int32) << 8) | (
            data[:, 2].astype(np.int32) << 16
        )
        values = (values ^ 0x800000) - 0x800000
        samples = values.astype(np.float32) / 8_388_608
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
    return samples


def load_recording(archive: zipfile.ZipFile, wav_name: str) -> dict:
    lab_name = wav_name[:-4] + ".lab"
    try:
        lab_text = archive.read(lab_name).decode("utf-8-sig")
    except KeyError as error:
        raise ValueError(f"Missing paired LAB file: {lab_name}") from error

    with wave.open(io.BytesIO(archive.read(wav_name)), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError(f"{wav_name}: only uncompressed PCM WAV is supported")
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        samples = decode_pcm(wav_file.readframes(wav_file.getnframes()), wav_file.getsampwidth())
    audio = samples.reshape(-1, channels).mean(axis=1)

    labels = []
    previous_end = 0
    for line_number, line in enumerate(lab_text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"{lab_name}:{line_number}: expected start end phoneme")
        try:
            start_tick, end_tick = int(fields[0]), int(fields[1])
        except ValueError as error:
            raise ValueError(f"{lab_name}:{line_number}: invalid timestamp") from error
        start = start_tick / TICKS_PER_SECOND
        end = end_tick / TICKS_PER_SECOND
        if start_tick < previous_end or end_tick <= start_tick:
            raise ValueError(f"{lab_name}:{line_number}: timestamps overlap or are empty")
        if end > len(audio) / sample_rate + 0.5:
            raise ValueError(f"{lab_name}:{line_number}: label extends beyond WAV duration")
        labels.append((start, end, fields[2]))
        previous_end = end_tick
    if not labels:
        raise ValueError(f"No phoneme labels found in {lab_name}")

    return {
        "name": pathlib.PurePosixPath(wav_name).name,
        "audio": audio,
        "sample_rate": sample_rate,
        "labels": labels,
    }


def magnitude_spectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    hop = max(1, round(sample_rate * 0.01))
    padded = np.pad(audio, (FFT_SIZE // 2, FFT_SIZE // 2))
    windows = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::hop]
    magnitudes = np.abs(np.fft.rfft(windows * np.hanning(FFT_SIZE), axis=1))
    max_bin = min(FFT_SIZE // 2 + 1, int(MAX_FREQUENCY * FFT_SIZE / sample_rate) + 1)
    return magnitudes[:, :max_bin].T.astype(np.float32)


def segment_magnitudes(
    audio: np.ndarray, sample_rate: int, start: float, end: float
) -> np.ndarray:
    centers = np.linspace(
        start * sample_rate, end * sample_rate, AVERAGE_FRAMES, endpoint=False
    ) + (end - start) * sample_rate / (2 * AVERAGE_FRAMES)
    indices = np.rint(centers).astype(np.int64)
    offsets = np.arange(FFT_SIZE) - FFT_SIZE // 2
    sample_indices = np.clip(indices[:, None] + offsets, 0, len(audio) - 1)
    windows = audio[sample_indices] * np.hanning(FFT_SIZE)
    magnitudes = np.abs(np.fft.rfft(windows, axis=1))
    max_bin = min(FFT_SIZE // 2 + 1, int(MAX_FREQUENCY * FFT_SIZE / sample_rate) + 1)
    return magnitudes[:, :max_bin].T.astype(np.float32)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if pathlib.Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def spectrogram_image(
    magnitudes: np.ndarray, reference: float, width: int, height: int
) -> Image.Image:
    db = 20 * np.log10(np.maximum(magnitudes / max(reference, 1e-12), 1e-8))
    intensity = np.clip((db + DYNAMIC_RANGE_DB) / DYNAMIC_RANGE_DB, 0, 1)
    stops = np.array(
        [[27, 12, 65], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float32,
    )
    position = intensity * (len(stops) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    rgb = (stops[lower] * (1 - fraction) + stops[upper] * fraction).astype(np.uint8)
    return Image.fromarray(rgb[::-1], "RGB").resize((width, height), Image.Resampling.BILINEAR)


def draw_output(recordings: list[dict], output: pathlib.Path) -> None:
    width = 3000
    left, right = 155, width - 70
    plot_width = right - left
    header_height = 160
    recording_height = 230
    average_top = header_height + recording_height * len(recordings) + 110
    phone_order = sorted({label[2] for record in recordings for label in record["labels"]})
    row_height = 48
    footer = 90
    height = average_top + len(phone_order) * row_height + footer
    image = Image.new("RGB", (width, height), "#101725")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    heading_font = load_font(23, bold=True)
    body_font = load_font(17)
    small_font = load_font(13)
    tiny_font = load_font(11)

    all_magnitudes = [record["magnitudes"] for record in recordings]
    reference = max(float(spectrum.max()) for spectrum in all_magnitudes)
    draw.text((left, 24), "Average phoneme spectrograms", font=title_font, fill="#f3f6fb")
    draw.text(
        (left, 82),
        f"{len(recordings)} WAV recordings | mean magnitude, time-normalized to "
        f"{AVERAGE_FRAMES} frames per phoneme | 0-{MAX_FREQUENCY // 1000} kHz",
        font=body_font,
        fill="#b9c5d7",
    )

    phone_sums: dict[str, np.ndarray] = {}
    phone_counts: dict[str, int] = {}
    phone_sources: dict[str, set[str]] = {}
    for index, record in enumerate(recordings):
        audio = record["audio"]
        sample_rate = record["sample_rate"]
        duration = len(audio) / sample_rate
        spec_top = header_height + index * recording_height
        spec_height = 150
        label_top = spec_top + spec_height + 5
        draw.text(
            (left, spec_top - 28),
            f'{record["name"]}  |  {duration:.2f} s  |  {len(record["labels"])} phonemes',
            font=heading_font,
            fill="#f3f6fb",
        )
        spec_image = spectrogram_image(record["magnitudes"], reference, plot_width, spec_height)
        image.paste(spec_image, (left, spec_top))

        for frequency in (0, 2000, 4000, 6000, 8000):
            y = spec_top + spec_height - round(frequency / MAX_FREQUENCY * spec_height)
            if y <= spec_top + spec_height:
                draw.line((left, y, right, y), fill="#59677a", width=1)
                draw.text((18, y - 9), f"{frequency // 1000}k", font=small_font, fill="#b9c5d7")

        for start, end, phone in record["labels"]:
            x0 = left + round(start / duration * plot_width)
            x1 = left + round(end / duration * plot_width)
            color_index = phone_order.index(phone)
            color = (
                75 + (color_index * 47) % 120,
                105 + (color_index * 67) % 120,
                130 + (color_index * 31) % 110,
            )
            draw.rectangle((x0, label_top, max(x0, x1 - 1), label_top + 23), fill=color)
            if x1 - x0 >= 18:
                label_width = draw.textbbox((0, 0), phone, font=tiny_font)[2]
                if label_width <= x1 - x0:
                    draw.text(
                        (x0 + (x1 - x0 - label_width) // 2, label_top + 5),
                        phone,
                        font=tiny_font,
                        fill="#ffffff",
                    )

            segment = segment_magnitudes(audio, sample_rate, start, end)
            if phone not in phone_sums:
                phone_sums[phone] = np.zeros_like(segment, dtype=np.float64)
                phone_counts[phone] = 0
                phone_sources[phone] = set()
            phone_sums[phone] += segment
            phone_counts[phone] += 1
            phone_sources[phone].add(record["name"])
        draw.text(
            (left, label_top + 27),
            "phoneme intervals  |  time →",
            font=small_font,
            fill="#b9c5d7",
        )

    draw.line((left, average_top - 22, right, average_top - 22), fill="#34445a", width=2)
    draw.text((left, average_top - 5), "Mean spectrum by phoneme", font=heading_font, fill="#f3f6fb")
    draw.text(
        (right - 630, average_top + 2),
        "n = occurrences | source WAVs     phoneme duration → 0% ... 100%",
        font=small_font,
        fill="#b9c5d7",
    )
    average_x = left + 390
    average_width = right - average_x
    for row, phone in enumerate(phone_order):
        y = average_top + 38 + row * row_height
        count = phone_counts[phone]
        mean = phone_sums[phone] / count
        tile = spectrogram_image(mean, reference, average_width, row_height - 5)
        image.paste(tile, (average_x, y))
        draw.text((left, y + 1), phone, font=heading_font, fill="#f3f6fb")
        source_names = ", ".join(
            pathlib.Path(name).stem for name in sorted(phone_sources[phone])
        )
        draw.text(
            (left + 65, y + 5),
            f"n={count} | {source_names}",
            font=small_font,
            fill="#b9c5d7",
        )
    draw.text(
        (left, height - 34),
        "Same color scale for all panels; average is computed in linear magnitude before dB conversion.",
        font=small_font,
        fill="#b9c5d7",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def main() -> None:
    args = parse_args()
    try:
        with zipfile.ZipFile(args.archive) as archive:
            wav_names = sorted(
                name for name in archive.namelist() if name.lower().endswith(".wav")
            )
            if not wav_names:
                raise ValueError(f"No WAV files found in {args.archive}")
            recordings = [load_recording(archive, name) for name in wav_names]
    except (OSError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Cannot read dataset archive: {error}") from error
    except ValueError as error:
        raise SystemExit(str(error)) from error

    for record in recordings:
        record["magnitudes"] = magnitude_spectrogram(record["audio"], record["sample_rate"])
    args.output = args.output.resolve()
    draw_output(recordings, args.output)
    print(f"Spectrogram image: {args.output}")
    print(f"Recordings: {len(recordings)}; phoneme classes: "
          f"{len({label[2] for record in recordings for label in record['labels']})}")


if __name__ == "__main__":
    main()
