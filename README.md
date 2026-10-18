# Liquid AI Audio Experiments

Command-line experiments built around Liquid AI's `LFM2.5-Audio-1.5B` model. The
main script runs the model locally for speech-to-speech experiments and records
transcripts, generated audio, and timing measurements.

## Features

- Run speech-to-speech on saved audio files or interactively through a microphone.
- Keep conversation context across live turns with `--multi-turn`.
- Adjust generation length, audio sampling settings, and retry suspicious or
  missing audio. Optionally re-speak the text reply with one of four built-in
  voices (`us_male`, `us_female`, `uk_male`, `uk_female`).
- Measure model startup, first-text/first-audio time, generation and decoding
  duration, real-time factor, and response audio duration. Results are written
  to CSV files under `outputs/`.
- Prepare evaluation manifests from audio folders or ZIP datasets, match
  transcripts from text/CSV/TSV files, and create short, resampled audio chunks.
- Evaluate ASR with word error rate (WER); optionally score response keywords
  and speech-output intelligibility using a speech-to-text round trip.
- Launch Liquid Audio's official Gradio demo, including a CPU compatibility
  wrapper for environments where the demo assumes CUDA.

## Repository contents

| File or folder | Purpose |
| --- | --- |
| `liquid_s2s.py` | Main saved-file and microphone speech-to-speech experiment |
| `chunk_audios.py` | Extract representative short audio clips and create a manifest |
| `prepare_dataset.py` | Create an evaluation manifest from an audio dataset |
| `liquid_accuracy.py` | Run ASR and optional speech-to-speech accuracy checks |
| `run_demo_cpu.py` | Launch the package's Gradio demo with a CPU `.cuda()` workaround |
| `accuracy_set.json`, `accuracy_en.json` | Example evaluation manifests |

The audio folders (`test_audio/`, `test_chunks_en/`, `audios_filtered/`) and the
example generated results (`outputs/`) are **not** stored in this repository.
They are hosted on Hugging Face (see below).

## Audio data

The audio files are too large for GitHub, so they are hosted as a Hugging Face
dataset:

https://huggingface.co/datasets/zaibun594/liquid-ai-audio

To download them into the project folder (run from the repository root):

```
pip install huggingface_hub
hf download zaibun594/liquid-ai-audio --repo-type dataset --local-dir .
```

This recreates `test_audio/`, `test_chunks_en/`, `audios_filtered/` and
`outputs/` where the scripts expect them.

> **Snapshot limitation:** this repository contains the files currently present
> in the project folder. The earlier `engine.py`, `llm.py`, `pipeline.py`, and
> `tests/` suite are not included. Consequently, `latency_live.py` (which imports
> those modules) and the earlier cascaded-pipeline tests cannot run from this
> snapshot. The standalone `liquid_s2s.py` workflow does not depend on them.

## Setup and tests

See [SETUP.md](SETUP.md) for clone, environment installation, syntax-check, and
model-inference commands. The inference check downloads the model weights on
first use and can be slow on CPU.

## Model and license

The model weights are downloaded from Hugging Face at runtime; they are not
stored in this repository. Review the model's terms and the
[Liquid Audio project](https://github.com/Liquid4All/liquid-audio) license
before use, especially for commercial use.