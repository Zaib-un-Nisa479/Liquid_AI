"""Liquid AI LFM2.5-Audio: speech-to-speech test (no separate LLM / ASR / TTS).

  python liquid_s2s.py live  [--turns 3] [--multi-turn]
  python liquid_s2s.py files  a.wav b.wav  or  a_folder/

Both modes print the model's text reply, save the reply WAV, and log latency to a CSV
(outputs/latency_live.csv and outputs/latency_files.csv, kept separate).

Experiment options (both modes):
  --system-prompt "..."        default is the official  "Respond with interleaved text and audio."
  --audio-temperature 1.0      official interleaved default (TTS mode uses 0.8)
  --audio-top-k 4              official interleaved default (TTS mode uses 64)
  --retries N                  regenerate up to N times if the reply audio looks distorted (heuristic)
  --tts-fallback               re-speak the reply TEXT with the model's TTS mode (slower, often steadier)
  --voice us_female            voice for --tts-fallback (us_male | us_female | uk_male | uk_female)

Env knobs: LFM_THREADS=8, LFM_DEVICE=cpu|cuda, LFM_AUDIO_REPO=<hf repo>
"""
import argparse
import csv
import os
import re
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# --- CPU-only fix: liquid_audio hardcodes .cuda() in its detokenizer ------------------
if not torch.cuda.is_available():
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality  # noqa: E402

HF_REPO = os.getenv("LFM_AUDIO_REPO", "LiquidAI/LFM2.5-Audio-1.5B")
DEVICE = os.getenv("LFM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == "cpu" and os.getenv("LFM_THREADS"):
    torch.set_num_threads(int(os.getenv("LFM_THREADS")))

OUT_SR = 24_000
MIC_SR = 16_000
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

DEFAULT_SYSTEM = "Respond with interleaved text and audio."
VOICES = {
    "us_male": "Perform TTS. Use the US male voice.",
    "us_female": "Perform TTS. Use the US female voice.",
    "uk_male": "Perform TTS. Use the UK male voice.",
    "uk_female": "Perform TTS. Use the UK female voice.",
}

METRICS = ["prep_ms", "first_text_ms", "first_audio_ms", "gen_ms", "decode_ms", "tts_fallback_ms",
           "response_ready_ms", "reply_audio_ms", "rtf", "attempts"]


# ------------------------------------------------------------------ model loading
def _load(cls):
    kwargs = {"device": DEVICE}
    if DEVICE == "cpu":
        kwargs["dtype"] = torch.float32
    for kw in (kwargs, {"device": DEVICE}, {}):
        try:
            return cls.from_pretrained(HF_REPO, **kw).eval()
        except TypeError:
            continue
    raise RuntimeError("could not load model; check liquid-audio version")


def load_model():
    t = time.perf_counter()
    print(f"[load] {HF_REPO} on {DEVICE} ...")
    processor = _load(LFM2AudioProcessor)
    model = _load(LFM2AudioModel)
    load_s = time.perf_counter() - t
    print(f"[load] done in {load_s:.1f}s")
    return processor, model, load_s


# ------------------------------------------------------------------ helpers
def strip_special(s):
    """Remove special tokens such as <|text_end|>."""
    return re.sub(r"<\|.*?\|>", "", s)


def audio_flag(wave):
    """Heuristic distortion check. Thresholds come from one observed bad clip
    (rms ~0.29, ~4% clipped samples vs. rms ~0.04 for normal replies): tune for your data."""
    if wave.size == 0:
        return "no_audio"
    clipped = float(np.mean(np.abs(wave) > 0.99))
    level = float(np.sqrt(np.mean(wave ** 2)))
    if clipped > 0.01 or level > 0.15:
        return "possible_distortion"
    return ""


def split_sentences(text):
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


@torch.no_grad()
def tts(processor, model, text, voice="us_female", temperature=0.8, top_k=64, max_new_tokens=512):
    """Sequential-mode TTS, sentence by sentence (the documented TTS settings)."""
    gap = np.zeros(int(0.15 * OUT_SR), dtype=np.float32)
    pieces = []
    for sentence in split_sentences(text):
        chat = ChatState(processor)
        chat.new_turn("system"); chat.add_text(VOICES[voice]); chat.end_turn()
        chat.new_turn("user"); chat.add_text(sentence); chat.end_turn()
        chat.new_turn("assistant")
        audio = []
        for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens,
                                           audio_temperature=temperature, audio_top_k=top_k):
            if t.numel() > 1:
                audio.append(t)
        if len(audio) > 1:
            codes = torch.stack(audio[:-1], 1).unsqueeze(0)
            pieces += [processor.decode(codes).cpu()[0].float().numpy(), gap]
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


# ------------------------------------------------------------------ core S2S call
@torch.no_grad()
def _s2s_once(processor, model, wav, sr, chat, max_new_tokens, system_prompt,
              audio_temperature, audio_top_k, tts_fallback, voice):
    t0 = time.perf_counter()
    if chat is None:
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(system_prompt)
        chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(wav, sr)
    chat.end_turn()
    chat.new_turn("assistant")
    t_prep = time.perf_counter()

    text_out, audio_out, modality = [], [], []
    t_first_text = t_first_audio = None
    print("Reply text: ", end="", flush=True)
    for t in model.generate_interleaved(
        **chat, max_new_tokens=max_new_tokens,
        audio_temperature=audio_temperature, audio_top_k=audio_top_k,
    ):
        now = time.perf_counter()
        if t.numel() == 1:
            if t_first_text is None:
                t_first_text = now
            text_out.append(t)
            modality.append(LFMModality.TEXT)
            print(strip_special(processor.text.decode(t)), end="", flush=True)
        else:
            if t_first_audio is None:
                t_first_audio = now
            audio_out.append(t)
            modality.append(LFMModality.AUDIO_OUT)
    t_gen_end = time.perf_counter()
    print()

    text = strip_special("".join(processor.text.decode(t) for t in text_out)).strip()

    tts_ms = decode_ms = ""
    if tts_fallback and text:
        print("[tts-fallback] re-speaking the reply text with TTS mode ...")
        wave = tts(processor, model, text, voice)
        tts_ms = round((time.perf_counter() - t_gen_end) * 1000, 1)
    else:
        wave = np.zeros(0, dtype=np.float32)
        if len(audio_out) > 1:
            codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)   # drop end-of-audio frame
            wave = processor.decode(codes).cpu()[0].float().numpy()
        decode_ms = round((time.perf_counter() - t_gen_end) * 1000, 1)

    if text_out and audio_out:
        chat.append(
            text=torch.stack(text_out, 1),
            audio_out=torch.stack(audio_out, 1),
            modality_flag=torch.tensor(modality),
        )
    chat.end_turn()

    def ms(a, b):
        return round((b - a) * 1000, 1) if a is not None and b is not None else ""

    metrics = {
        "prep_ms": ms(t0, t_prep),
        "first_text_ms": ms(t0, t_first_text),
        "first_audio_ms": ms(t0, t_first_audio),
        "gen_ms": ms(t_prep, t_gen_end),
        "decode_ms": decode_ms,
        "tts_fallback_ms": tts_ms,
        "reply_audio_ms": round(len(wave) / OUT_SR * 1000, 1),
    }
    return text, wave, chat, metrics


def s2s(processor, model, wav, sr, chat=None, max_new_tokens=512, system_prompt=DEFAULT_SYSTEM,
        audio_temperature=1.0, audio_top_k=4, tts_fallback=False, voice="us_female", retries=0):
    """wav: float tensor [1, T]. Returns (text, wave_np, chat, metrics).
    Retries only apply to single-turn calls (chat=None)."""
    t_all = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        text, wave, chat_out, m = _s2s_once(processor, model, wav, sr, chat, max_new_tokens, system_prompt,
                                            audio_temperature, audio_top_k, tts_fallback, voice)
        flag = audio_flag(wave)
        if flag and attempt <= retries and chat is None:
            print(f"[check] {flag} -> retrying ({attempt}/{retries}) ...")
            continue
        break
    done = time.perf_counter()
    audio_s = len(wave) / OUT_SR
    m["response_ready_ms"] = round((done - t_all) * 1000, 1)     # includes any retries
    m["rtf"] = round((done - t_all) / audio_s, 3) if audio_s else ""
    m["attempts"] = attempt
    m["audio_flag"] = flag
    if flag:
        print(f"[check] WARNING: reply audio flagged as {flag}")
    return text, wave, chat_out, m


def gen_kwargs(a):
    return dict(max_new_tokens=a.max_new_tokens, system_prompt=a.system_prompt,
                audio_temperature=a.audio_temperature, audio_top_k=a.audio_top_k,
                tts_fallback=a.tts_fallback, voice=a.voice, retries=a.retries)


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def print_summary(title, rows, extra_cols=()):
    steady = rows[1:] if len(rows) > 1 else rows
    note = "excluding first warm-up run" if len(rows) > 1 else "single run"
    print(f"\n=== {title}: median latency ({note}) ===")
    for col in list(extra_cols) + METRICS:
        vals = [float(r[col]) for r in steady if r.get(col, "") != ""]
        if vals:
            print(f"{col:<32}: {statistics.median(vals):.1f}")
    flagged = sum(1 for r in rows if r.get("audio_flag"))
    print(f"{'replies flagged (audio check)':<32}: {flagged}/{len(rows)}")


def print_row(m):
    print(f"[latency] first_text {m['first_text_ms']} ms | first_audio {m['first_audio_ms']} ms | "
          f"gen {m['gen_ms']} ms | decode {m['decode_ms']} ms | tts_fallback {m['tts_fallback_ms']} ms | "
          f"ready {m['response_ready_ms']} ms | reply audio {m['reply_audio_ms']} ms | RTF {m['rtf']} | "
          f"attempts {m['attempts']}")


# ------------------------------------------------------------------ MODE 1: live mic
def record_until_enter():
    import sounddevice as sd
    chunks = []

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    cmd = input("\nPress Enter to START recording (or type q + Enter to quit): ").strip().lower()
    if cmd == "q":
        return None, None
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", callback=cb):
        input("Recording... speak, then press Enter to STOP.")
        t_end = time.perf_counter()
    if not chunks:
        return np.zeros(0, dtype=np.float32), t_end
    return np.concatenate(chunks).reshape(-1), t_end


def run_live(args):
    import sounddevice as sd
    processor, model, load_s = load_model()
    live_dir = OUT_DIR / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    rows, chat = [], None

    for i in range(1, args.turns + 1):
        print(f"\n---------- Live turn {i}/{args.turns} ----------")
        audio, t_end = record_until_enter()
        if audio is None:
            break
        if audio.size < MIC_SR * 0.3:
            print("Recording too short, skipping.")
            continue
        wav = torch.from_numpy(audio.reshape(1, -1).copy())

        text, wave, chat_out, m = s2s(processor, model, wav, MIC_SR,
                                      chat=chat if args.multi_turn else None, **gen_kwargs(args))
        if args.multi_turn:
            chat = chat_out

        t_play = None
        if wave.size:
            sd.play(wave, OUT_SR)
            t_play = time.perf_counter()
        in_path = live_dir / f"turn_{i:02d}_input.wav"
        out_path = live_dir / f"turn_{i:02d}_reply.wav"
        sf.write(str(in_path), audio, MIC_SR)
        if wave.size:
            sf.write(str(out_path), wave, OUT_SR)
            sd.wait()
        else:
            print("!! Model produced no audio for this turn.")

        m["capture_end_to_playback_ms"] = round((t_play - t_end) * 1000, 1) if t_play else ""
        rows.append({"turn": i, "warmup": i == 1, "input_audio_ms": round(len(audio) / MIC_SR * 1000, 1),
                     "reply_text": text, **m})
        print(f"Saved: {in_path.name}, {out_path.name}")
        print_row(m)
        print(f"[latency] speech-end -> playback start: {m['capture_end_to_playback_ms']} ms  <-- what you feel")

    cols = ["turn", "warmup", "input_audio_ms", "reply_text"] + METRICS + ["audio_flag", "capture_end_to_playback_ms"]
    csv_path = OUT_DIR / "latency_live.csv"
    if rows:
        write_csv(csv_path, rows, cols)
        print_summary("LIVE MIC", rows, extra_cols=["capture_end_to_playback_ms"])
        print(f"\nModel load time: {load_s:.1f}s | CSV: {csv_path}")


# ------------------------------------------------------------------ MODE 2: saved files
def collect_files(paths):
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        elif p.exists():
            files.append(p)
        else:
            print(f"!! not found: {p}")
    return files


def run_files(args):
    files = collect_files(args.paths)
    if not files:
        raise SystemExit("No input audio files found.")
    processor, model, load_s = load_model()
    files_dir = OUT_DIR / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, f in enumerate(files, 1):
        print(f"\n---------- File {i}/{len(files)}: {f.name} ----------")
        data, sr = sf.read(str(f), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        text, wave, _, m = s2s(processor, model, wav, sr, chat=None, **gen_kwargs(args))
        out_path = files_dir / f"{f.stem}__reply.wav"
        if wave.size:
            sf.write(str(out_path), wave, OUT_SR)
            print(f"Saved: {out_path}")
        else:
            print("!! Model produced no audio for this file.")
        print_row(m)
        rows.append({"file": f.name, "warmup": i == 1, "input_audio_ms": round(wav.shape[1] / sr * 1000, 1),
                     "reply_text": text, **m})

    cols = ["file", "warmup", "input_audio_ms", "reply_text"] + METRICS + ["audio_flag"]
    csv_path = OUT_DIR / "latency_files.csv"
    write_csv(csv_path, rows, cols)
    print_summary("SAVED FILES", rows)
    print(f"\nModel load time: {load_s:.1f}s | CSV: {csv_path}")


# ------------------------------------------------------------------ CLI
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-new-tokens", type=int, default=512)
    common.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    common.add_argument("--audio-temperature", type=float, default=1.0)
    common.add_argument("--audio-top-k", type=int, default=4)
    common.add_argument("--retries", type=int, default=0)
    common.add_argument("--tts-fallback", action="store_true")
    common.add_argument("--voice", choices=sorted(VOICES), default="us_female")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    lv = sub.add_parser("live", parents=[common], help="microphone in, speaker out")
    lv.add_argument("--turns", type=int, default=3)
    lv.add_argument("--multi-turn", action="store_true", help="keep conversation history across turns")
    lv.set_defaults(fn=run_live)

    fl = sub.add_parser("files", parents=[common], help="saved audio files in, saved audio out")
    fl.add_argument("paths", nargs="+", help="audio files and/or folders")
    fl.set_defaults(fn=run_files)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
