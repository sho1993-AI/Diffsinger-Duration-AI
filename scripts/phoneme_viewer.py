#!/usr/bin/env python3
"""Serve a local browser UI for inspecting training and inference phonemes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


ROOT_DIR = (
    pathlib.Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else pathlib.Path(__file__).resolve().parents[1]
)
TICKS_PER_SECOND = 10_000_000
WORKER_SCRIPTS = {
    "predict_duration.py",
    "predict_sho_duration.py",
    "synthesize_checkpoint.py",
    "synthesize_onnx.py",
}


def worker_command(script_name: str, *arguments: str | pathlib.Path) -> list[str]:
    if script_name not in WORKER_SCRIPTS:
        raise ValueError(f"Unknown worker script: {script_name}")
    if getattr(sys, "frozen", False):
        return [sys.executable, "--_worker", script_name, *(str(arg) for arg in arguments)]
    venv_dir = pathlib.Path(os.environ.get("VENV_DIR", ROOT_DIR / ".venv-cpu312")).resolve()
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise ValueError(f"CPU Python environment not found: {python}")
    return [str(python), str(ROOT_DIR / "scripts" / script_name), *(str(arg) for arg in arguments)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--training-archive", type=pathlib.Path, default=ROOT_DIR / "data/dataset.zip")
    parser.add_argument(
        "--inference-dir", type=pathlib.Path,
        default=ROOT_DIR / "data",
    )
    parser.add_argument("--sho-archive", type=pathlib.Path, default=ROOT_DIR / "data/sho2.zip")
    return parser.parse_args()


def read_lab(text: str, name: str) -> list[dict]:
    labels = []
    previous_end = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"{name}:{line_number}: expected start end phoneme")
        try:
            start_tick, end_tick = int(fields[0]), int(fields[1])
        except ValueError as error:
            raise ValueError(f"{name}:{line_number}: timestamps must be integers") from error
        if start_tick < previous_end or end_tick <= start_tick:
            raise ValueError(f"{name}:{line_number}: timestamps overlap or are empty")
        labels.append({
            "phoneme": fields[2],
            "start": start_tick / TICKS_PER_SECOND,
            "end": end_tick / TICKS_PER_SECOND,
            "duration": (end_tick - start_tick) / TICKS_PER_SECOND,
        })
        previous_end = end_tick
    if not labels:
        raise ValueError(f"No phoneme labels found in {name}")
    return labels


def parse_ds_segment(segment: dict) -> dict:
    phonemes = str(segment.get("ph_seq", "")).split()
    durations = str(segment.get("ph_dur", "")).split()
    notes = str(segment.get("note_seq", "")).split()
    note_durations = str(segment.get("note_dur", "")).split()
    try:
        phone_durations = [float(value) for value in durations]
        note_lengths = [float(value) for value in note_durations]
        offset = float(segment.get("offset", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid numeric duration or offset in .ds data") from error
    return {
        "offset": offset,
        "text": str(segment.get("text", "")),
        "phonemes": [
            {"phoneme": phone, "duration": phone_durations[index] if index < len(phone_durations) else None}
            for index, phone in enumerate(phonemes)
        ],
        "notes": [
            {"note": note, "duration": note_lengths[index] if index < len(note_lengths) else None}
            for index, note in enumerate(notes)
        ],
        "duration_mismatch": len(phonemes) != len(phone_durations),
    }


def parse_inference_file(filename: str, content: str) -> dict:
    suffix = pathlib.PurePath(filename).suffix.lower()
    if suffix == ".lab":
        labels = read_lab(content, filename)
        return {
            "name": filename,
            "segments": [{
                "offset": 0,
                "text": "",
                "phonemes": [
                    {"phoneme": label["phoneme"], "duration": label["duration"]}
                    for label in labels
                ],
                "notes": [],
                "duration_mismatch": False,
            }],
        }
    if suffix not in {".ds", ".json"}:
        raise ValueError("対応形式は .ds / .json / .lab です")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON parse error: {error}") from error
    if isinstance(data, dict):
        data = data.get("segments", [data])
    if not isinstance(data, list) or not data:
        raise ValueError(".ds data must contain at least one segment")
    if not all(isinstance(segment, dict) for segment in data):
        raise ValueError("Each .ds segment must be an object")
    return {"name": filename, "segments": [parse_ds_segment(segment) for segment in data]}


def training_records(archive_path: pathlib.Path) -> list[dict]:
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {member.filename for member in archive.infolist() if not member.is_dir()}
        wav_names = sorted(name for name in members if name.lower().endswith(".wav"))
        for wav_name in wav_names:
            lab_name = str(pathlib.PurePosixPath(wav_name).with_suffix(".lab"))
            if lab_name not in members:
                continue
            labels = read_lab(archive.read(lab_name).decode("utf-8-sig"), lab_name)
            with wave.open(io.BytesIO(archive.read(wav_name)), "rb") as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
            records.append({
                "id": wav_name,
                "name": pathlib.PurePosixPath(wav_name).stem,
                "audio": f"/api/audio?name={wav_name}",
                "duration": duration,
                "labels": labels,
            })
    if not records:
        raise ValueError(f"No paired WAV/LAB files found in {archive_path}")
    return records


def inference_samples(inference_dir: pathlib.Path) -> list[str]:
    if not inference_dir.is_dir():
        return []
    samples = []
    seen_content = set()
    for path in sorted(inference_dir.rglob("*.ds")):
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest in seen_content:
            continue
        seen_content.add(digest)
        samples.append(path.relative_to(inference_dir).as_posix())
    return samples


def duration_models(diffsinger_dir: pathlib.Path) -> list[dict]:
    models = []
    for experiment_dir in sorted((diffsinger_dir / "checkpoints").glob("duration_cpu_*")):
        if not (experiment_dir / "config.yaml").is_file():
            continue
        checkpoints = []
        for path in experiment_dir.glob("model_ckpt_steps_*.ckpt"):
            match = re.fullmatch(r"model_ckpt_steps_(\d+)\.ckpt", path.name)
            if match:
                checkpoints.append(int(match.group(1)))
        if checkpoints:
            dictionary_path = experiment_dir / "dictionary-ja.txt"
            phonemes = []
            if dictionary_path.is_file():
                phonemes = [
                    line.split()[0]
                    for line in dictionary_path.read_text(encoding="utf-8-sig").splitlines()
                    if line.split()
                ]
            models.append({
                "experiment": experiment_dir.name,
                "checkpoint": max(checkpoints),
                "phonemes": phonemes,
            })
    return models


def acoustic_models(diffsinger_dir: pathlib.Path) -> list[dict]:
    models = []
    checkpoints_dir = diffsinger_dir / "checkpoints"
    for experiment_dir in sorted(checkpoints_dir.iterdir() if checkpoints_dir.is_dir() else []):
        if experiment_dir.name.startswith("duration_cpu_"):
            continue
        config_path = experiment_dir / "config.yaml"
        if not config_path.is_file():
            continue
        config = config_path.read_text(encoding="utf-8-sig")
        if "training.acoustic_task.AcousticTask" not in config:
            continue

        vocoder_line = next(
            (line.split(":", maxsplit=1)[1] for line in config.splitlines()
             if line.lstrip().startswith("vocoder_ckpt:")),
            "",
        )
        vocoder_checkpoint = vocoder_line.split("#", maxsplit=1)[0].strip().strip("'\"")
        if not vocoder_checkpoint:
            base_config = diffsinger_dir / "configs/acoustic.yaml"
            if base_config.is_file():
                for line in base_config.read_text(encoding="utf-8-sig").splitlines():
                    if line.lstrip().startswith("vocoder_ckpt:"):
                        vocoder_checkpoint = line.split(":", maxsplit=1)[1].split("#", maxsplit=1)[0].strip().strip("'\"")
                        break
        if not vocoder_checkpoint or not (diffsinger_dir / vocoder_checkpoint).is_file():
            continue

        checkpoints = []
        for path in experiment_dir.glob("model_ckpt_steps_*.ckpt"):
            match = re.fullmatch(r"model_ckpt_steps_(\d+)\.ckpt", path.name)
            if match:
                checkpoints.append(int(match.group(1)))
        if not checkpoints:
            continue

        speaker_map_path = experiment_dir / "spk_map.json"
        speakers = []
        if speaker_map_path.is_file():
            speakers = list(json.loads(speaker_map_path.read_text(encoding="utf-8-sig")))
        models.append({
            "experiment": experiment_dir.name,
            "format": "checkpoint",
            "checkpoint": max(checkpoints),
            "speakers": speakers,
            "requires_speaker": bool(re.search(r"(?m)^use_spk_id:\s*true\s*$", config)),
            "label": f"{experiment_dir.name} · step {max(checkpoints)}",
        })

    vocoder_checkpoint = (
        diffsinger_dir / "checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt"
    )
    if not vocoder_checkpoint.is_file():
        return models

    artifact_roots = [diffsinger_dir / "artifacts", ROOT_DIR / "data/acoustic_models"]
    for artifact_root in artifact_roots:
        if not artifact_root.is_dir():
            continue
        for model_path in sorted(artifact_root.rglob("*.onnx")):
            package_dir = model_path.parent
            config_path = package_dir / "dsconfig.yaml"
            if not config_path.is_file():
                continue
            config = config_path.read_text(encoding="utf-8-sig")
            model_name = re.search(r"(?m)^acoustic:\s*['\"]?([^'\"#\s]+)", config)
            phonemes_name = re.search(r"(?m)^phonemes:\s*['\"]?([^'\"#\s]+)", config)
            if not model_name or not phonemes_name or model_name.group(1) != model_path.name:
                continue
            phonemes_path = (package_dir / phonemes_name.group(1)).resolve()
            if package_dir.resolve() not in phonemes_path.parents or not phonemes_path.is_file():
                continue
            languages_name = re.search(r"(?m)^languages:\s*['\"]?([^'\"#\s]+)", config)
            if not languages_name:
                continue
            languages_path = (package_dir / languages_name.group(1)).resolve()
            if package_dir.resolve() not in languages_path.parents or not languages_path.is_file():
                continue
            package_settings = {
                key: re.search(rf"(?m)^{key}:\s*['\"]?([^'\"#\s]+)", config)
                for key in ("sample_rate", "hop_size", "num_mel_bins", "mel_base")
            }
            if any(not match for match in package_settings.values()):
                continue
            if (
                package_settings["sample_rate"].group(1) != "44100"
                or package_settings["hop_size"].group(1) != "512"
                or package_settings["num_mel_bins"].group(1) != "128"
                or package_settings["mel_base"].group(1) != "e"
            ):
                continue
            package_id = f"onnx:{artifact_root.name}:{model_path.relative_to(artifact_root).as_posix()}"
            speaker_match = re.search(
                r"(?m)^speakers:\s*\n((?:[ \t]+-[ \t]*[^\n]+\n?)+)",
                config,
            )
            speakers = []
            if speaker_match:
                speakers = [
                    line.split("-", maxsplit=1)[1].strip().strip("'\"")
                    for line in speaker_match.group(1).splitlines()
                    if line.strip().startswith("-")
                ]
            models.append({
                "experiment": package_id,
                "format": "onnx",
                "checkpoint": None,
                "speakers": speakers,
                "requires_speaker": bool(speakers),
                "label": f"ONNX · {package_dir.name}/{model_path.name}",
                "_model_path": str(model_path.resolve()),
                "_package_dir": str(package_dir.resolve()),
            })
    return models


def inference_models(diffsinger_dir: pathlib.Path, sho_archive: pathlib.Path) -> list[dict]:
    models = []
    if sho_archive.is_file():
        with zipfile.ZipFile(sho_archive) as archive:
            phonemes = sorted(json.loads(archive.read("sho2/phonemes.json")))
        models.append({
            "experiment": "sho2_pretrained",
            "label": f"SHO pretrained ONNX · {len(phonemes)}音素",
            "checkpoint": None,
            "phonemes": phonemes,
            "pretrained": True,
        })
    recommended = None
    reports = sorted(
        ROOT_DIR.glob("duration_sweep_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if reports:
        try:
            recommended = json.loads(reports[0].read_text(encoding="utf-8")).get("best", {}).get("experiment")
        except (OSError, json.JSONDecodeError):
            recommended = None

    raw_labels = ROOT_DIR / "data/duration_cpu/raw_mora/transcriptions.csv"
    observed_phones = set()
    if raw_labels.is_file():
        with raw_labels.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                observed_phones.update(row["ph_seq"].split())
    for model in duration_models(diffsinger_dir):
        untrained = set(model["phonemes"]) - observed_phones
        if "AP" in untrained and "SP" in observed_phones:
            untrained.remove("AP")
        model["untrained_phonemes"] = sorted(untrained)
        model["recommended"] = model["experiment"] == recommended
        prefix = "Recommended · " if model["recommended"] else ""
        model["label"] = (
            f"{prefix}{model['experiment']} · step {model['checkpoint']} · "
            f"{len(model['untrained_phonemes'])}未学習音素"
        )
        models.append(model)
    return models


def predict_duration(
    content: str,
    filename: str,
    experiment: str,
    checkpoint: int,
    diffsinger_dir: pathlib.Path,
) -> dict:
    if pathlib.PurePath(filename).suffix.lower() not in {".ds", ".json"}:
        raise ValueError("Duration推論には .ds / .json ファイルが必要です")
    parsed = parse_inference_file(filename, content)
    available = {model["experiment"]: model for model in duration_models(diffsinger_dir)}
    if experiment not in available or available[experiment]["checkpoint"] < checkpoint:
        raise ValueError("選択した学習済みcheckpointが見つかりません")
    supported_phones = set(available[experiment]["phonemes"])
    unsupported_phones = sorted({
        phone["phoneme"]
        for segment in parsed["segments"]
        for phone in segment["phonemes"]
        if phone["phoneme"] not in supported_phones
    })
    if unsupported_phones:
        raise ValueError(f"学習辞書にない音素のため推論できません: {' '.join(unsupported_phones)}")

    with tempfile.TemporaryDirectory(prefix="duration-infer-") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        input_path = temp_path / "input.ds"
        output_dir = temp_path / "output"
        input_path.write_text(content, encoding="utf-8")
        command = worker_command(
            "predict_duration.py",
            str(input_path), str(output_dir),
            "--experiment", experiment,
            "--checkpoint", str(checkpoint),
            "--diffsinger-dir", str(diffsinger_dir),
        )
        try:
            result = subprocess.run(
                command,
                cwd=diffsinger_dir,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("Duration推論が5分でタイムアウトしました") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-1800:].strip()
            raise ValueError(f"Duration推論に失敗しました: {detail}")
        output_path = output_dir / "duration_prediction.ds"
        predicted_source = output_path.read_text(encoding="utf-8-sig")
        predicted = parse_inference_file(
            pathlib.Path(filename).stem + ".ds",
            predicted_source,
        )
        calibration = re.search(r"Calibrated (\d+) leading phonemes", result.stdout)
    predicted["model"] = experiment
    predicted["checkpoint"] = checkpoint
    predicted["calibrated_leading_count"] = int(calibration.group(1)) if calibration else 0
    predicted["leading_duration_cap"] = 0.3
    predicted["source"] = predicted_source
    return predicted


def predict_sho_duration(content: str, filename: str, sho_archive: pathlib.Path) -> dict:
    if pathlib.PurePath(filename).suffix.lower() not in {".ds", ".json"}:
        raise ValueError("Duration推論には .ds / .json ファイルが必要です")
    parse_inference_file(filename, content)
    with tempfile.TemporaryDirectory(prefix="sho-duration-infer-") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        input_path = temp_path / "input.ds"
        output_path = temp_path / "predicted.ds"
        input_path.write_text(content, encoding="utf-8")
        command = worker_command(
            "predict_sho_duration.py",
            str(input_path), str(output_path), "--archive", str(sho_archive),
        )
        try:
            result = subprocess.run(
                command,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("SHO Duration推論が5分でタイムアウトしました") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-1800:].strip()
            raise ValueError(f"SHO Duration推論に失敗しました: {detail}")
        predicted_source = output_path.read_text(encoding="utf-8-sig")
        predicted = parse_inference_file(
            pathlib.Path(filename).stem + ".ds",
            predicted_source,
        )
        calibration = re.search(r"Calibrated (\d+) leading phonemes", result.stdout)
    predicted["model"] = "SHO pretrained ONNX"
    predicted["checkpoint"] = None
    predicted["calibrated_leading_count"] = int(calibration.group(1)) if calibration else 0
    predicted["leading_duration_cap"] = 0.3
    predicted["source"] = predicted_source
    return predicted


def prepare_synthesis(
    content: str,
    filename: str,
    experiment: str,
    checkpoint: int | None,
    speaker: str,
    diffsinger_dir: pathlib.Path,
    job_dir: pathlib.Path,
) -> tuple[list[str], pathlib.Path, pathlib.Path]:
    if pathlib.PurePath(filename).suffix.lower() not in {".ds", ".json"}:
        raise ValueError("合成には .ds / .json ファイルが必要です")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON parse error: {error}") from error
    segments = data if isinstance(data, list) else data.get("segments", [data]) if isinstance(data, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ValueError(".ds data must contain at least one segment")
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f".ds segment {index} must be an object")
        missing = [key for key in ("ph_seq", "ph_dur", "f0_seq", "f0_timestep") if not segment.get(key)]
        if missing:
            raise ValueError(f"セグメント{index}に {', '.join(missing)} がありません。先にDuration推論を実行してください")
        if len(str(segment["ph_seq"]).split()) != len(str(segment["ph_dur"]).split()):
            raise ValueError(f"セグメント{index}のph_seqとph_durの数が一致しません")

    model = next(
        (item for item in acoustic_models(diffsinger_dir) if item["experiment"] == experiment),
        None,
    )
    if model is None or (
        model["format"] == "checkpoint"
        and (checkpoint is None or model["checkpoint"] < checkpoint)
    ):
        raise ValueError("選択した音響モデルまたはvocoderが見つかりません")
    if speaker and speaker not in model["speakers"]:
        raise ValueError("選択した歌声がモデルにありません")
    if model["requires_speaker"] and len(model["speakers"]) > 1 and not speaker:
        if not all(isinstance(segment.get("spk_mix"), dict) and segment["spk_mix"] for segment in segments):
            raise ValueError("複数歌手モデルではGUIで歌声を選ぶか、.dsにspk_mixを指定してください")

    output_dir = job_dir / "output"
    stream_dir = job_dir / "segments"
    input_path = job_dir / "input.ds"
    output_path = output_dir / "synthesis.wav"
    input_path.write_text(content, encoding="utf-8")
    if model["format"] == "onnx":
        command = worker_command(
            "synthesize_onnx.py",
            str(input_path), str(output_path),
            "--model", model["_model_path"],
            "--package-dir", model["_package_dir"],
            "--diffsinger-dir", str(diffsinger_dir),
            "--stream-dir", str(stream_dir),
        )
        if speaker:
            command.extend(["--speaker", speaker])
    else:
        command = worker_command(
            "synthesize_checkpoint.py",
            str(input_path), str(output_path),
            "--diffsinger-dir", str(diffsinger_dir),
            "--experiment", experiment,
            "--checkpoint", str(checkpoint),
            "--stream-dir", str(stream_dir),
        )
        if speaker:
            command.extend(["--speaker", speaker])
    return command, output_path, stream_dir


def create_handler(
    training_archive: pathlib.Path,
    inference_dir: pathlib.Path,
    diffsinger_dir: pathlib.Path,
    sho_archive: pathlib.Path,
):
    synthesis_jobs: dict[str, dict] = {}
    synthesis_jobs_lock = threading.Lock()

    def run_synthesis_job(
        job_id: str,
        job: dict,
        command: list[str],
        output_path: pathlib.Path,
        stream_dir: pathlib.Path,
        diffsinger_dir: pathlib.Path,
    ) -> None:
        def append_log(message: str) -> None:
            with synthesis_jobs_lock:
                job["logs"].append(message)
                del job["logs"][:-100]

        process: subprocess.Popen | None = None

        def consume_output() -> None:
            assert process is not None and process.stdout is not None
            for line in process.stdout:
                text = line.strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    append_log(text)
                    continue
                if not isinstance(event, dict):
                    append_log(text)
                    continue
                if event.get("type") == "segment":
                    with synthesis_jobs_lock:
                        job["segments"].append({
                            "index": int(event["index"]),
                            "total": int(event["total"]),
                            "label": str(event["label"]),
                            "duration": float(event["duration"]),
                            "offset": float(event["offset"]),
                        })
                    append_log(
                        f"フレーズ {event['index']}/{event['total']} を合成しました: {event['label']}"
                    )
                elif event.get("type") == "log":
                    append_log(str(event.get("message", "")))
                else:
                    append_log(text)

        try:
            with synthesis_jobs_lock:
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                    job["finished_at"] = time.monotonic()
                    return
            append_log("合成ジョブを開始しました")
            process = subprocess.Popen(
                command,
                cwd=diffsinger_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with synthesis_jobs_lock:
                job["process"] = process
                cancel_requested = job["cancel_requested"]
            if cancel_requested:
                process.terminate()
            reader = threading.Thread(target=consume_output, daemon=True)
            reader.start()
            try:
                return_code = process.wait(timeout=900)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait()
                reader.join()
                raise ValueError("音声合成が15分でタイムアウトしました") from error
            reader.join()
            with synthesis_jobs_lock:
                cancelled = job["cancel_requested"]
            if cancelled:
                append_log("合成をキャンセルしました")
                with synthesis_jobs_lock:
                    job["status"] = "cancelled"
                    job["finished_at"] = time.monotonic()
                return
            if return_code != 0:
                with synthesis_jobs_lock:
                    detail = "\n".join(job["logs"][-12:]).strip()
                raise ValueError(detail or f"DiffSinger exited with status {return_code}")
            if not output_path.is_file():
                raise ValueError("DiffSingerの合成音声が見つかりません")
            try:
                with wave.open(str(output_path), "rb") as wav_file:
                    if wav_file.getnframes() == 0:
                        raise ValueError("合成音声が空です")
            except (wave.Error, EOFError) as error:
                raise ValueError("DiffSingerの出力が有効なWAVではありません") from error
            append_log("全フレーズの合成が完了しました")
            with synthesis_jobs_lock:
                job["status"] = "completed"
                job["finished_at"] = time.monotonic()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            append_log(str(error))
            with synthesis_jobs_lock:
                job["status"] = "failed"
                job["error"] = str(error)
                job["finished_at"] = time.monotonic()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} - {format % args}")

        def send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            request = urlparse(self.path)
            query = parse_qs(request.query)
            try:
                if request.path == "/":
                    body = (ROOT_DIR / "scripts/phoneme_viewer.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if request.path == "/api/training":
                    self.send_json({"records": training_records(training_archive)})
                    return
                if request.path == "/api/audio":
                    name = query.get("name", [""])[0]
                    with zipfile.ZipFile(training_archive) as archive:
                        if name not in archive.namelist() or not name.lower().endswith(".wav"):
                            self.send_error(404)
                            return
                        body = archive.read(name)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if request.path == "/api/inference-samples":
                    self.send_json({"samples": inference_samples(inference_dir)})
                    return
                if request.path == "/api/models":
                    models = acoustic_models(diffsinger_dir)
                    self.send_json({
                        "models": inference_models(diffsinger_dir, sho_archive),
                        "synthesis_models": [
                            {key: value for key, value in model.items() if not key.startswith("_")}
                            for model in models
                        ],
                    })
                    return
                if request.path == "/api/synthesis-status":
                    job_id = query.get("job", [""])[0]
                    with synthesis_jobs_lock:
                        job = synthesis_jobs.get(job_id)
                        if job is None:
                            self.send_error(404)
                            return
                        payload = {
                            "status": job["status"],
                            "logs": list(job["logs"]),
                            "segments": list(job["segments"]),
                            "total": job["total"],
                            "error": job["error"],
                        }
                    self.send_json(payload)
                    return
                if request.path in {"/api/synthesis-segment", "/api/synthesis-audio"}:
                    job_id = query.get("job", [""])[0]
                    with synthesis_jobs_lock:
                        job = synthesis_jobs.get(job_id)
                        if job is None:
                            self.send_error(404)
                            return
                        if request.path == "/api/synthesis-segment":
                            try:
                                index = int(query.get("index", ["0"])[0])
                            except ValueError:
                                self.send_error(404)
                                return
                            if index < 1 or index > len(job["segments"]):
                                self.send_error(404)
                                return
                            audio_path = job["stream_dir"] / f"segment-{index:06d}.wav"
                        elif job["status"] == "completed":
                            audio_path = job["output_path"]
                        else:
                            audio_path = job["stream_dir"] / "partial.wav"
                        if not audio_path.is_file():
                            self.send_json({"error": "音声データはまだ準備中です"}, 202)
                            return
                    body = audio_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if request.path == "/api/inference-sample":
                    relative_name = query.get("name", [""])[0]
                    candidate = (inference_dir / relative_name).resolve()
                    if inference_dir.resolve() not in candidate.parents or candidate.suffix.lower() != ".ds":
                        self.send_error(404)
                        return
                    content = candidate.read_text(encoding="utf-8-sig")
                    payload = parse_inference_file(candidate.name, content)
                    payload["source"] = content
                    self.send_json(payload)
                    return
                self.send_error(404)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self) -> None:
            request_path = urlparse(self.path).path
            if request_path == "/api/synthesis-cancel":
                query = parse_qs(urlparse(self.path).query)
                job_id = query.get("job", [""])[0]
                with synthesis_jobs_lock:
                    job = synthesis_jobs.get(job_id)
                    if job is None:
                        self.send_error(404)
                        return
                    if job["status"] != "running":
                        self.send_json({"cancelled": False}, 200)
                        return
                    job["cancel_requested"] = True
                    process = job.get("process")
                if process is not None and process.poll() is None:
                    process.terminate()
                self.send_json({"cancelled": True}, 202)
                return
            if request_path not in {"/api/inference", "/api/predict-duration", "/api/synthesize"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 5_000_000:
                    raise ValueError("File must be between 1 byte and 5 MB")
                filename = pathlib.PurePath(self.headers.get("X-Filename", "inference.ds")).name
                content = self.rfile.read(length).decode("utf-8-sig")
                if request_path == "/api/inference":
                    self.send_json(parse_inference_file(filename, content))
                    return
                if request_path == "/api/synthesize":
                    experiment = self.headers.get("X-Model", "")
                    model = next(
                        (item for item in acoustic_models(diffsinger_dir)
                         if item["experiment"] == experiment),
                        None,
                    )
                    if model is None:
                        raise ValueError("音響モデルとvocoderのcheckpointをDiffSingerに配置してください")
                    try:
                        checkpoint = (
                            int(self.headers.get("X-Checkpoint", model["checkpoint"]))
                            if model["format"] == "checkpoint" else None
                        )
                    except ValueError as error:
                        raise ValueError("checkpointの指定が不正です") from error
                    with synthesis_jobs_lock:
                        now = time.monotonic()
                        expired = [
                            job_id for job_id, job in synthesis_jobs.items()
                            if job["status"] != "running"
                            and now - job["finished_at"] > 1800
                        ]
                        for old_job_id in expired:
                            shutil.rmtree(synthesis_jobs[old_job_id]["root"], ignore_errors=True)
                            del synthesis_jobs[old_job_id]
                        if any(job["status"] == "running" for job in synthesis_jobs.values()):
                            self.send_json({"error": "別の合成ジョブが実行中です"}, 429)
                            return
                    job_id = uuid.uuid4().hex
                    job_root = pathlib.Path(tempfile.mkdtemp(prefix="diffsinger-synthesis-"))
                    try:
                        command, output_path, stream_dir = prepare_synthesis(
                            content,
                            filename,
                            experiment,
                            checkpoint,
                            self.headers.get("X-Speaker", ""),
                            diffsinger_dir,
                            job_root,
                        )
                    except (OSError, ValueError):
                        shutil.rmtree(job_root, ignore_errors=True)
                        raise
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as error:
                        shutil.rmtree(job_root, ignore_errors=True)
                        raise ValueError(f"JSON parse error: {error}") from error
                    segments = parsed if isinstance(parsed, list) else parsed.get("segments", [parsed])
                    job = {
                        "status": "running",
                        "root": job_root,
                        "stream_dir": stream_dir,
                        "output_path": output_path,
                        "logs": [],
                        "segments": [],
                        "total": len(segments),
                        "error": None,
                        "cancel_requested": False,
                        "process": None,
                    }
                    with synthesis_jobs_lock:
                        synthesis_jobs[job_id] = job
                    thread = threading.Thread(
                        target=run_synthesis_job,
                        args=(job_id, job, command, output_path, stream_dir, diffsinger_dir),
                        daemon=True,
                    )
                    thread.start()
                    self.send_json({"job_id": job_id}, 202)
                    return
                experiment = self.headers.get("X-Model", "")
                if experiment == "sho2_pretrained":
                    self.send_json(predict_sho_duration(content, filename, sho_archive))
                    return
                model = next((item for item in duration_models(diffsinger_dir) if item["experiment"] == experiment), None)
                if model is None:
                    raise ValueError("学習済みDurationモデルが選択されていません")
                self.send_json(predict_duration(
                    content,
                    filename,
                    experiment,
                    model["checkpoint"],
                    diffsinger_dir,
                ))
            except (OSError, UnicodeDecodeError, ValueError) as error:
                self.send_json({"error": str(error)}, 400)

    return Handler


def main() -> None:
    args = parse_args()
    training_archive = args.training_archive.resolve()
    inference_dir = args.inference_dir.resolve()
    diffsinger_dir = pathlib.Path(os.environ.get("DIFFSINGER_DIR", ROOT_DIR / "third_party/DiffSinger")).resolve()
    sho_archive = args.sho_archive.resolve()
    if not training_archive.is_file():
        raise SystemExit(f"Training archive not found: {training_archive}")
    training_records(training_archive)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_handler(training_archive, inference_dir, diffsinger_dir, sho_archive),
    )
    print(f"Phoneme viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Training data: {training_archive}", flush=True)
    print(f"Inference samples: {inference_dir}", flush=True)
    if getattr(sys, "frozen", False):
        threading.Timer(0.8, webbrowser.open, args=(f"http://{args.host}:{args.port}/",)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping phoneme viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        if len(sys.argv) < 3 or sys.argv[2] not in WORKER_SCRIPTS:
            raise SystemExit("Invalid packaged worker command")
        worker_path = ROOT_DIR / "scripts" / sys.argv[2]
        if not worker_path.is_file():
            raise SystemExit(f"Packaged worker script not found: {worker_path}")
        sys.path.insert(0, str(worker_path.parent))
        sys.argv = [str(worker_path), *sys.argv[3:]]
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True, write_through=True)
        runpy.run_path(str(worker_path), run_name="__main__")
    elif len(sys.argv) > 1 and sys.argv[1] == "--_self-test-runtime":
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        if "CPUExecutionProvider" not in providers:
            raise SystemExit("The bundled ONNX Runtime CPU provider is missing")
        print(f"ONNX Runtime {onnxruntime.__version__}; providers: {', '.join(providers)}")
    else:
        main()
