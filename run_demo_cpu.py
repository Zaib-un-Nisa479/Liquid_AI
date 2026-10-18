"""Launch Liquid's official Gradio demo with the CPU-only .cuda() workaround.

  python run_demo_cpu.py        # then open http://localhost:7860

Use this only if plain `liquid-audio-demo` crashes with "Torch not compiled with CUDA enabled".
It calls the same entry point the package registers (liquid_audio.demo.chat:main).
"""
import torch

if not torch.cuda.is_available():
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self

from liquid_audio.demo.chat import main  # noqa: E402

if __name__ == "__main__":
    main()
