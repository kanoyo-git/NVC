# NVC Studio — FAQ

## Getting started

**What is NVC Studio?**
A local web console for the NVC voice-conversion toolkit: inference, stem separation, model library, training, and checkpoint tools in one page. It runs on your machine; audio never leaves it unless you import models from a link.

**Where do I put voice models?**
Drop `.pth` files into the `assets/weights` folder and press **Refresh voices** on the Inference page. Index files (`.index`) are picked up automatically from `logs/`.

## Inference

**What does the index do?**
A trained Faiss index mixes retrieval-based timbre into the result. **Index rate** controls how strongly it is applied (0 = off, 1 = full). Use it to preserve the target voice's character on long vowels.

**Which pitch method should I use?**
`rmvpe` is the default and works well for singing. `fcpe` is faster and good for speech. `pm` is a lightweight fallback when the others misbehave.

**What do Envelope mix and Consonant protect do?**
Envelope mix blends the source volume envelope into the output (higher = closer to the source dynamics). Consonant protect keeps unvoiced consonants from being altered by the index and pitch shifts.

**Transpose values?**
Semitones. `+12` is one octave up, `-12` one octave down. Match the source singer to the target voice's range.

## Separation

**What does the Separate page do?**
It splits audio into two stems with the bundled pymss models: vocals/accompaniment, dereverb, or main-vocal extraction. The "aggressive" variants cut deeper but may cost quality.

## Training

**Recommended workflow?**
1. Training page — set the experiment name, sample rate, version.
2. Step 2a — slice and normalize a dataset folder or a dropped `.zip`.
3. Step 2b — extract HuBERT and pitch features.
4. Step 3 — train the model, then train the index.

**How much data do I need?**
10–30 minutes of clean, single-speaker audio is a good start. More helps, but noisy data hurts more than less data.

**Multi-speaker training?**
Choose the multiple-speakers dataset type and either name subfolders `Name_ID_Repeat` (IDs 0–109) or submit a manifest on the Speakers page.

## Library and checkpoints

**What can I import?**
Model archives (`.zip`) from your device or from a link (Drive, Mega, Yandex, MediaFire, Hugging Face), and pretrained v2 G/D pairs for fine-tuning.

**What are the checkpoint tools for?**
Merge two models with a weight, edit embedded model info, inspect a checkpoint's metadata, and extract a small inference-only model from a training checkpoint.

## Troubleshooting

**The device readout shows CPU.**
CUDA was not detected. Install a CUDA-enabled PyTorch build from the matching `requirments_cu*.txt` file.

**Conversion fails with an index error.**
The selected `.index` file does not match the model or is missing. Pick **No index** or retrain the index for that voice.
