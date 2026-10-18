"""Measure turn-based microphone -> STT -> dummy LLM -> TTS latency.

Usage: python latency_live.py [turns] [record_seconds]
"""
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from engine import AudioEngine, OUT_SR
from llm import LLM
from pipeline import VoicePipeline


INPUT_SR = 16_000
VOICE = "us_female"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CSV_PATH = OUTPUT_DIR / "latency.csv"


class DummyLLM(LLM):
    def chat(self, messages: list[dict]) -> str:
        user_text = next(
            (message["content"] for message in reversed(messages)
             if message["role"] == "user"),
            "",
        )
        return f"I heard you say: {user_text}"


def ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def record_voice_turn(record_seconds: float, sr: int = INPUT_SR,
                     threshold: float = 0.02,
                     silence_after: float = 0.35,
                     max_wait: float = 8.0):
    """Wait until real speech is detected, then capture audio and stop after a quiet gap."""
    block_size = max(256, int(sr * 0.05))
    chunks = []
    started = False
    silence_time = 0.0
    total_time = 0.0
    speech_hold = 0.0

    stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32")
    stream.start()
    print("Listening...")
    base_values = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.6:
        chunk, _ = stream.read(block_size)
        if chunk.size == 0:
            continue
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        base_values.append(float(np.sqrt(np.mean(samples ** 2))))

    noise_floor = float(np.median(base_values)) if base_values else 0.0
    trigger = max(threshold, noise_floor * 5.0 + 0.015)
    print(f"Mic noise floor: {noise_floor:.4f}, trigger: {trigger:.4f}")
    t0 = time.perf_counter()

    try:
        while time.perf_counter() - t0 < max_wait:
            chunk, _ = stream.read(block_size)
            if chunk.size == 0:
                continue
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
            chunk_seconds = len(samples) / sr
            total_time += chunk_seconds

            if not started:
                if rms >= trigger:
                    speech_hold += chunk_seconds
                    if speech_hold >= 0.20:
                        started = True
                        print("Speech detected. Recording...")
                        chunks.append(samples)
                        silence_time = 0.0
                else:
                    speech_hold = 0.0
                continue

            chunks.append(samples)
            if rms >= trigger:
                silence_time = 0.0
            else:
                silence_time += chunk_seconds

            if total_time >= record_seconds and silence_time >= silence_after:
                break

            if total_time >= record_seconds:
                break

            if len(chunks) > 0 and total_time > 0.5 and silence_time >= silence_after:
                break
    finally:
        stream.stop(); stream.close()

    if not chunks:
        return np.zeros(int(sr * record_seconds), dtype=np.float32)

    audio = np.concatenate(chunks)
    if audio.size > int(sr * record_seconds):
        audio = audio[:int(sr * record_seconds)]
    if audio.size < int(sr * 0.25):
        pad = np.zeros(int(sr * 0.25), dtype=np.float32)
        pad[:audio.size] = audio
        return pad
    return audio.astype(np.float32)


def main() -> None:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    record_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    if turns < 1 or record_seconds <= 0:
        raise SystemExit("turns must be >= 1 and record_seconds must be > 0")

    OUTPUT_DIR.mkdir(exist_ok=True)
    load_started = time.perf_counter()
    engine = AudioEngine()
    pipeline = VoicePipeline(engine=engine, llm=DummyLLM(), voice=VOICE)
    model_load_ms = ms(time.perf_counter() - load_started)
    print(f"Model startup: {model_load_ms:.3f} ms")
    print(f"Each turn records about {record_seconds:g}s after speech starts.")

    columns = [
        "turn", "warmup", "transcript", "reply", "capture_ms", "stt_ms",
        "llm_ms", "tts_ms", "pipeline_wall_ms", "capture_end_to_reply_ready_ms",
        "record_start_to_reply_ready_ms", "playback_ms", "full_turn_ms",
        "reply_audio_ms", "tts_rtf",
    ]
    steady_turns = []

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        for turn_number in range(1, turns + 1):
            print(f"\nTurn {turn_number}/{turns}")
            turn_started = time.perf_counter()
            recorded = record_voice_turn(record_seconds, sr=INPUT_SR)
            capture_finished = time.perf_counter()

            input_path = OUTPUT_DIR / f"mic_turn_{turn_number:02d}.wav"
            sf.write(str(input_path), recorded, INPUT_SR)
            wav = torch.from_numpy(recorded.reshape(1, -1).copy())

            pipeline_started = time.perf_counter()
            result = pipeline.run(wav, INPUT_SR)
            reply_ready = time.perf_counter()

            playback_started = time.perf_counter()
            if result.reply_audio.size:
                sd.play(result.reply_audio, OUT_SR)
                sd.wait()
            playback_finished = time.perf_counter()

            capture_s = capture_finished - turn_started
            pipeline_wall_s = reply_ready - pipeline_started
            capture_to_reply_s = reply_ready - capture_finished
            record_to_reply_s = reply_ready - turn_started
            playback_s = playback_finished - playback_started
            full_turn_s = playback_finished - turn_started
            audio_s = result.audio_seconds
            tts_rtf = result.t_tts / audio_s if audio_s else 0.0

            row = {
                "turn": turn_number,
                "warmup": turn_number == 1,
                "transcript": result.transcript,
                "reply": result.reply_text,
                "capture_ms": ms(capture_s),
                "stt_ms": ms(result.t_stt),
                "llm_ms": ms(result.t_llm),
                "tts_ms": ms(result.t_tts),
                "pipeline_wall_ms": ms(pipeline_wall_s),
                "capture_end_to_reply_ready_ms": ms(capture_to_reply_s),
                "record_start_to_reply_ready_ms": ms(record_to_reply_s),
                "playback_ms": ms(playback_s),
                "full_turn_ms": ms(full_turn_s),
                "reply_audio_ms": ms(audio_s),
                "tts_rtf": round(tts_rtf, 4),
            }
            writer.writerow(row)
            csv_file.flush()

            if turn_number > 1:
                steady_turns.append(row)

            print(f"You: {result.transcript}")
            print(f"Reply: {result.reply_text}")
            print(
                f"STT {row['stt_ms']:.3f} ms | LLM {row['llm_ms']:.3f} ms | "
                f"TTS {row['tts_ms']:.3f} ms | "
                f"capture end -> reply ready {row['capture_end_to_reply_ready_ms']:.3f} ms"
            )

    if steady_turns:
        print("\nMedian latency excluding first (warm-up) turn:")
        for column in (
            "stt_ms", "llm_ms", "tts_ms", "pipeline_wall_ms",
            "capture_end_to_reply_ready_ms", "record_start_to_reply_ready_ms",
            "full_turn_ms",
        ):
            value = statistics.median(float(row[column]) for row in steady_turns)
            print(f"{column}: {value:.3f} ms")

    print(f"\nPer-turn measurements: {CSV_PATH}")


if __name__ == "__main__":
    main()