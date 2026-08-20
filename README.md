<div align="center">

[English](./README.md) | [Русский](./README.ru.md)

# NVC

### Retrieval-based voice conversion, training and vocal-production toolkit

[![License: MIT](https://img.shields.io/badge/license-MIT-2f6f4e.svg)](./LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-356f9f.svg)](https://www.python.org/downloads/)
[![Interface: Studio](https://img.shields.io/badge/interface-NVC%20Studio-6f4eb5.svg)](#run-nvc)

[![Colab: English](https://img.shields.io/badge/Colab-English-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kanoyo-git/NVC/blob/main/NVC.ipynb)
[![Colab: Русский](https://img.shields.io/badge/Colab-%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kanoyo-git/NVC/blob/main/NVC.ru.ipynb)

[Overview](#what-is-nvc) · [Features](#features) · [Quick start](#quick-start) · [Dynamic autotune](#dataset-aware-dynamic-autotune) · [Credits](#credits-and-acknowledgements)

</div>

## What is NVC?

NVC is a local voice-conversion framework based on the RVC approach. It transforms the timbre of a source voice into a trained target voice while retaining the original performance's phrasing, rhythm and articulation.

The system combines pitch extraction, neural voice synthesis and retrieval from the target model's training features. Retrieval helps reduce source-timbre leakage and makes the converted result sound closer to the selected voice. NVC also includes model training, batch inference, vocal separation, real-time voice conversion and a dataset-aware dynamic autotune designed for singing voices.

NVC runs locally. Your recordings and models stay on your machine unless you choose to upload or publish them elsewhere.

## Features

| Area | What NVC provides |
| --- | --- |
| Voice conversion | Single-file and batch inference with optional feature-index retrieval |
| Pitch processing | RMVPE, FCPE and PM extraction, manual transposition and dataset-aware dynamic autotune |
| Model training | Single-speaker and multi-speaker workflows, preprocessing, feature extraction and checkpoint tools |
| Interfaces | Modern NVC Studio, legacy Gradio GUI, offline CLI and a real-time voice changer |
| Vocal production | pymss/MSST-based vocal and accompaniment separation |
| Model tools | Index handling, checkpoint extraction and model merging |
| Hardware | NVIDIA CUDA acceleration; CPU fallback for AMD and Intel, with DirectML support on Windows |

The original RVC guidance still applies to training data: at least 10 minutes of clean, low-noise speech or isolated vocals is recommended. More varied, well-recorded material generally produces a more stable model.

## Quick start

This branch targets **64-bit Python 3.12**. Run the commands from the repository root. Ubuntu 24.04 x86_64 is the recommended Linux environment.

### 1. Clone the repository

```bash
git clone https://github.com/kanoyo-git/NVC.git
cd NVC
```

### 2. Create a virtual environment

Ubuntu 24.04:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies for your hardware

CPU, AMD or Intel:

```bash
python -m pip install -r requirments_cpu_py312.txt
```

NVIDIA RTX 50 series:

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

NVIDIA GPUs before the RTX 50 series:

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

Verify the installation:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

The requirement files use mainland-China mirrors by default. If needed, replace only their `--index-url` and `--extra-index-url` values with the official PyPI and PyTorch indexes; keep package versions, CUDA suffixes and the two-stage installation order unchanged.

### 4. Download runtime models

The required upstream assets are hosted in the [VoiceConversionWebUI model repository](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main).

```bash
python -m pip install --upgrade huggingface_hub

# Required for inference and feature extraction
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe

# Required for v1/v2 training
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pretrained/*" "pretrained_v2/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
python -m zipfile -e .model-downloads/mute.zip logs

# Required only for pymss/MSST vocal separation
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pymss_weights/*" --local-dir assets
```

Windows AMD/Intel DirectML environments additionally need:

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

On Windows, if FFmpeg is not available system-wide, place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true) in the repository root.

## Models and directories

NVC creates runtime directories automatically. Keep user models and indexes in the following locations:

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── rmvpe/
│   └── rmvpe.pt
├── pretrained/
├── pretrained_v2/
├── pymss_weights/
├── weights/          # user .pth voice models
└── indices/          # user .index files
logs/
└── mute/             # silence samples used during training
```

## Run NVC

Start the modern NVC Studio:

```bash
python gui.py
```

The interface opens in a browser and listens on port `7865` by default. On a headless server, use:

```bash
python gui.py --noautoopen
```

Start the previous Gradio interface:

```bash
python gui.py --legacy
```

Start the real-time voice changer:

```bash
python realtime_gui.py
```

Windows launchers are also included: `go-gui.bat` and `go-realtime_gui.bat`.

### Offline CLI

```bash
python -m infer.cli \
  --model assets/weights/voice.pth \
  --input input.flac \
  --output output.flac \
  --f0-method rmvpe
```

Use `python -m infer.cli --help` for batch conversion, recursive directory scanning, speaker selection and output-format options.

## Dataset-aware dynamic autotune

The dynamic autotune adapts a conversion to the target model's vocal range without blindly transposing the entire song. It analyzes the target dataset, stores a reusable pitch profile beside the model and processes the source vocal phrase by phrase. Octave decisions are smoothed over time so that register changes do not tear sustained notes, disturb vibrato or alter the song's timing.

In NVC Studio or the legacy GUI:

1. Select a voice model.
2. Enable **Dynamic autotune**.
3. Choose a dataset directory or provide one representative audio file.
4. Build the profile, then process the vocal.

The generated sidecar is named `MODEL.pitch.json` and is reused during later conversions.

Build a profile and convert in one CLI command:

```bash
python -m infer.cli \
  --model assets/weights/voice.pth \
  --input song.flac \
  --output converted.flac \
  --pitch-profile-dataset /path/to/voice-dataset \
  --autotune
```

Or build the profile separately:

```bash
python -m tools.build_pitch_profile \
  /path/to/voice-dataset \
  --model assets/weights/voice.pth
```

## Credits and acknowledgements

NVC is built on the work of the [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and its contributors. The retrieval-based voice-conversion architecture, training workflow and much of the original application foundation come from that project.

The inherited RVC base models were trained with nearly 50 hours of high-quality audio from the open VCTK dataset. NVC preserves the original project's acknowledgement of that dataset and its authors.

Core projects and research used by or carried forward into NVC:

- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [RMVPE](https://github.com/Dream-High/RMVPE)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [pymss](https://github.com/pymss-project/pymss)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [Gradio](https://github.com/gradio-app/gradio)
- [FastAPI](https://github.com/fastapi/fastapi)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)

The RMVPE pretrained model was trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss), as credited by the original project.

Thank you to every person who has reported issues, submitted fixes, tested models, translated documentation or contributed research and code:

- [NVC contributors](https://github.com/kanoyo-git/NVC/graphs/contributors)
- [RVC contributors](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors)

## License and responsible use

NVC is distributed under the [MIT License](./LICENSE). Third-party components and models may have their own licenses; review them before redistribution or commercial use.

Use only recordings, datasets and voice models for which you have the necessary rights and consent. You are responsible for audio produced or distributed with this software.
