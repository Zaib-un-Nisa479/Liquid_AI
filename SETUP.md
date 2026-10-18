# Setup and test

## Clone the repository

Install [Git](https://git-scm.com/) first. In PowerShell:

```powershell
git clone https://github.com/Zaib-un-Nisa479/Liquid_AI.git
cd Liquid_AI
```

## Create an environment (Windows PowerShell)

Python 3.12 is recommended. Install a PyTorch build that matches your hardware
from [the PyTorch install selector](https://pytorch.org/get-started/locally/),
then install the project requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install torch and torchaudio for your CPU or GPU, following pytorch.org.
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, replace the CPU PyTorch command with the CUDA command
recommended by the PyTorch install selector. The CPU command is only appropriate
for CPU-only installations.

## Download the audio data

The audio files are hosted on Hugging Face because they are too large for
GitHub. From the repository root, run:

```powershell
python -m pip install huggingface_hub
hf download zaibun594/liquid-ai-audio --repo-type dataset --local-dir .
```

This creates `test_audio/`, `test_chunks_en/`, `audios_filtered/` and
`outputs/` in the project folder. If `hf` is not recognised on an older
install, use `huggingface-cli download` with the same arguments.

## Check the code and run an inference test

Run a syntax check without downloading model weights:

```powershell
python -m compileall -q chunk_audios.py latency_live.py liquid_accuracy.py liquid_s2s.py prepare_dataset.py run_demo_cpu.py
```

Run speech-to-speech on the included sample to test the installed dependencies
and model (this needs the audio data downloaded above):

```powershell
python liquid_s2s.py files --max-new-tokens 128 test_audio/p1__clean.wav
```

The first inference downloads the `LiquidAI/LFM2.5-Audio-1.5B` weights, which
requires several gigabytes of disk space. CPU inference can take a while. The
generated reply and timing results are saved under `outputs/`.

## Other commands

Run a microphone session (requires a working microphone and speaker):

```powershell
python liquid_s2s.py live --turns 3 --multi-turn
```

Evaluate an existing manifest:

```powershell
python liquid_accuracy.py accuracy_set.json
python liquid_accuracy.py accuracy_set.json --s2s
```

The `--s2s` option also generates a reply and evaluates its speech round trip,
so it takes longer. Update the manifest's audio paths and reference text to
match the clips you want to evaluate.

To launch the optional Gradio demo, install its extra dependencies and run:

```powershell
python -m pip install "liquid-audio[demo]"
python run_demo_cpu.py
```