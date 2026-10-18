"""Accuracy tests for LFM2.5-Audio (put next to liquid_s2s.py).

  python liquid_accuracy.py accuracy_set.json            # 1) ASR word error rate only (fastest)
  python liquid_accuracy.py accuracy_set.json --s2s      # + 2) answer keywords + 3) speech round-trip

Manifest (JSON list):
  [{"audio": "outputs/live/turn_01_input.wav",
    "reference": "how does machine learning work",     # exactly what was said (for ASR WER)
    "keywords": ["data", "learn"]}]                      # words a correct answer must contain (optional)

Metrics
  1. ASR WER        : model transcribes your audio vs. your reference text.       lower = better
  2. Answer keywords: S2S reply text contains all expected keywords.              higher = better
  3. Round-trip WER : model's spoken reply is transcribed again and compared to
                      the reply text it intended to say (speech intelligibility).  lower = better
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import liquid_s2s as L          # applies the CPU fix and gives us load_model() / s2s()
from liquid_audio import ChatState

OUT = L.OUT_DIR / "accuracy"


def clean(text):
    return re.sub(r"<\|.*?\|>", "", text).strip()


def normalize(text):
    text = re.sub(r"[^\w\s']", " ", clean(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def load_audio(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.mean(axis=1)).unsqueeze(0), sr


@torch.no_grad()
def asr(processor, model, wav, sr, max_new_tokens=256):
    chat = ChatState(processor)
    chat.new_turn("system"); chat.add_text("Perform ASR."); chat.end_turn()
    chat.new_turn("user"); chat.add_audio(wav, sr); chat.end_turn()
    chat.new_turn("assistant")
    out = []
    for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
        if t.numel() == 1:
            out.append(processor.text.decode(t))
    return clean("".join(out))


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--s2s", action="store_true", help="also run speech-to-speech (slow on CPU)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    processor, model, _ = L.load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, it in enumerate(items, 1):
        path = Path(it["audio"])
        print(f"\n---------- Item {i}/{len(items)}: {path.name} ----------")
        wav, sr = load_audio(path)
        row = {"audio": path.name, "reference": it.get("reference", ""), "asr_text": "", "asr_wer": None,
               "reply_text": "", "keywords": ",".join(it.get("keywords", [])), "keywords_ok": None,
               "roundtrip_text": "", "roundtrip_wer": None}

        # 1) ASR accuracy
        hyp = asr(processor, model, wav, sr)
        row["asr_text"] = hyp
        print(f"ASR heard : {hyp}")
        if it.get("reference"):
            row["asr_wer"] = round(wer(it["reference"], hyp), 3)
            print(f"ASR WER   : {row['asr_wer']:.1%}  (reference: {it['reference']})")

        # 2) + 3) answer correctness and speech-output intelligibility
        if args.s2s:
            text, wave, _, _ = L.s2s(processor, model, wav, sr, chat=None, max_new_tokens=args.max_new_tokens)
            text = clean(text)
            row["reply_text"] = text
            if it.get("keywords"):
                row["keywords_ok"] = all(k.lower() in text.lower() for k in it["keywords"])
                print(f"Keywords  : {'PASS' if row['keywords_ok'] else 'FAIL'} {it['keywords']}")
            if wave.size:
                sf.write(str(OUT / f"{path.stem}__reply.wav"), wave, L.OUT_SR)
                back = asr(processor, model, torch.from_numpy(wave.reshape(1, -1).copy()), L.OUT_SR)
                row["roundtrip_text"] = back
                row["roundtrip_wer"] = round(wer(text, back), 3)
                print(f"Round-trip: heard back \"{back}\"")
                print(f"Round-trip WER: {row['roundtrip_wer']:.1%}")
            else:
                print("!! no reply audio produced")
        rows.append(row)

    csv_path = OUT / "accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== ACCURACY SUMMARY ===")
    print(f"Items tested         : {len(rows)}")
    print(f"Mean ASR WER         : {mean([r['asr_wer'] for r in rows])}  (0 = perfect)")
    kw = [r["keywords_ok"] for r in rows if r["keywords_ok"] is not None]
    if kw:
        print(f"Answer keyword pass  : {sum(kw)}/{len(kw)}")
    rt = mean([r["roundtrip_wer"] for r in rows])
    if rt is not None:
        print(f"Mean round-trip WER  : {rt}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
