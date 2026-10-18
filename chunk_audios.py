"""Cut short test chunks out of long audio files (CPU-friendly) and build a manifest.

  python chunk_audios.py audios.zip                       # 1 chunk of 12 s per file
  python chunk_audios.py audios.zip --chunks-per-file 2 --chunk-seconds 10

For each file it tries several positions, keeps the loudest ones (so you don't test silence),
saves them as mono 16 kHz WAVs in test_chunks/, and writes accuracy_set.json for
liquid_accuracy.py. "reference" is left empty: type what is really said in each chunk
(only for clips you want WER scored on).
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

EXT = {".wav", ".flac", ".ogg", ".mp3"}
TARGET_SR = 16_000
POSITIONS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]   # candidate start points (fraction of file)


def read_segment(path, sr, start_s, length_s):
    data, _ = sf.read(str(path), start=int(start_s * sr), frames=int(length_s * sr),
                      dtype="float32", always_2d=True)
    return data.mean(axis=1)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help=".zip or folder of audio files")
    ap.add_argument("--chunk-seconds", type=float, default=12.0)
    ap.add_argument("--chunks-per-file", type=int, default=1)
    ap.add_argument("--extract-to", default="audios_extracted")
    ap.add_argument("--out-dir", default="test_chunks")
    ap.add_argument("--out", default="accuracy_set.json")
    args = ap.parse_args()

    src = Path(args.dataset)
    if src.suffix.lower() == ".zip":
        root = Path(args.extract_to)
        if not root.exists():
            print(f"Extracting {src} -> {root} ...")
            with zipfile.ZipFile(src) as z:
                z.extractall(root)
        else:
            print(f"Using existing folder {root}")
    else:
        root = src

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)
    stems_wav = {p.stem for p in files if p.suffix.lower() == ".wav"}
    files = [p for p in files if not (p.suffix.lower() == ".mp3" and p.stem in stems_wav)]  # skip duplicate mp3
    if not files:
        raise SystemExit("No audio files found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    print(f"\n{'#':>2}  {'duration':>9}  {'chunks':>6}  file")

    for n, p in enumerate(files, 1):
        try:
            info = sf.info(str(p))
            dur, sr = info.duration, info.samplerate
            L = min(args.chunk_seconds, dur)
            if dur <= args.chunk_seconds:
                starts = [0.0]
            else:
                cands = [pos * (dur - L) for pos in POSITIONS]
                scored = sorted(((rms(read_segment(p, sr, s, L)), s) for s in cands), reverse=True)
                starts = sorted(s for _, s in scored[: args.chunks_per_file])
            saved = 0
            for k, s in enumerate(starts, 1):
                mono = read_segment(p, sr, s, L)
                wav = AF.resample(torch.from_numpy(mono), sr, TARGET_SR).numpy() if sr != TARGET_SR else mono
                name = f"{n:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', p.stem)[:40]}__c{k}.wav"
                sf.write(str(out_dir / name), wav, TARGET_SR, subtype="PCM_16")
                items.append({"audio": (out_dir / name).as_posix(), "reference": "", "keywords": [],
                              "source": p.name, "start_s": round(s, 1), "duration_s": round(L, 1)})
                saved += 1
            print(f"{n:>2}  {dur / 60:>7.1f} m  {saved:>6}  {p.name}")
        except Exception as e:
            print(f"{n:>2}  SKIPPED ({e})  {p.name}")

    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(items)} chunks -> {out_dir}/   manifest -> {args.out}")
    print("Next: python liquid_accuracy.py", args.out)


if __name__ == "__main__":
    main()
