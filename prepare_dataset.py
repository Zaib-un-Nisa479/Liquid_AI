"""Turn a dataset ZIP (or folder) into a manifest for liquid_accuracy.py.

  python prepare_dataset.py dataset.zip --limit 20 --max-seconds 15
  python prepare_dataset.py my_folder  --limit 10 --out accuracy_small.json

Finds transcripts from (any of):
  - a .txt with the same name as each audio file          (clip1.wav + clip1.txt)
  - LibriSpeech style  *.trans.txt   (lines: "<clip-id> <text>")
  - a .csv/.tsv with a path/file/audio column and a text/sentence/transcript column
"""
import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

import soundfile as sf

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
PATH_COLS = ["path", "file", "filename", "file_name", "audio", "audio_path", "audio_filepath", "clip", "id"]
TEXT_COLS = ["text", "sentence", "transcript", "transcription", "reference", "normalized_text"]


def read(p):
    return p.read_text(encoding="utf-8", errors="ignore")


def find_references(root):
    refs = {}
    for t in root.rglob("*.txt"):
        if t.name.endswith(".trans.txt"):
            for line in read(t).splitlines():
                if " " in line:
                    key, text = line.split(" ", 1)
                    refs[key.strip()] = text.strip()
        else:
            refs.setdefault(t.stem, read(t).strip())
    for m in list(root.rglob("*.csv")) + list(root.rglob("*.tsv")):
        delim = "\t" if m.suffix.lower() == ".tsv" else ","
        with m.open(encoding="utf-8", errors="ignore", newline="") as f:
            rd = csv.DictReader(f, delimiter=delim)
            cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
            pc = next((cols[c] for c in PATH_COLS if c in cols), None)
            tc = next((cols[c] for c in TEXT_COLS if c in cols), None)
            if pc and tc:
                for r in rd:
                    if r.get(pc) and r.get(tc):
                        refs[Path(r[pc]).stem] = r[tc].strip()
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="path to .zip or a folder")
    ap.add_argument("--limit", type=int, default=20, help="how many clips to sample")
    ap.add_argument("--max-seconds", type=float, default=15.0, help="skip clips longer than this (CPU is slow)")
    ap.add_argument("--extract-to", default="dataset")
    ap.add_argument("--out", default="accuracy_set.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.dataset)
    if src.suffix.lower() == ".zip":
        root = Path(args.extract_to)
        print(f"Extracting {src} -> {root} ...")
        with zipfile.ZipFile(src) as z:
            z.extractall(root)
    else:
        root = src

    audio = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    refs = find_references(root)
    print(f"Audio files found : {len(audio)}")
    print(f"Transcripts found : {len(refs)}")
    if not audio:
        raise SystemExit("No audio files found (wav/flac/ogg/mp3).")

    random.Random(args.seed).shuffle(audio)
    items, skipped = [], 0
    for p in audio:
        try:
            dur = sf.info(str(p)).duration
        except Exception:
            skipped += 1          # unreadable (e.g. mp3 on an old libsndfile)
            continue
        if dur > args.max_seconds or dur < 0.5:
            skipped += 1
            continue
        item = {"audio": p.as_posix()}
        ref = refs.get(p.stem)
        if ref:
            item["reference"] = ref
        items.append(item)
        if len(items) >= args.limit:
            break

    with_ref = sum(1 for i in items if "reference" in i)
    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(items)} clips ({with_ref} with transcripts, {skipped} skipped) -> {args.out}")
    if with_ref == 0:
        print("!! No transcripts matched. ASR WER can't be scored; tell me the dataset layout "
              "and I'll adapt the matching.")


if __name__ == "__main__":
    main()
