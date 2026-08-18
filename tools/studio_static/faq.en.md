# NVC Studio — FAQ

## Getting started

**What is NVC Studio?**
A local web console for the NVC voice-conversion toolkit. It bundles inference, stem separation, a model library, training and checkpoint tools into a single page that runs in your browser. Everything stays on your machine — audio is only sent out when you import a model from a link.

**Where do I put voice models?**
Drop `.pth` files into `assets/weights` and press **Refresh voices** on the Inference page. Index files (`.index`) are picked up automatically from `logs/`.

## Inference

**What does the index do?**
A trained Faiss index adds retrieval-based timbre to the result — the model looks up the nearest known audio fragments and mixes them in. **Index rate** controls how strong that effect is: 0 turns it off, 1 applies it fully. Useful for keeping the target voice's character on long sustained notes.

**Which pitch method should I use?**
`rmvpe` is the default and works well for singing. `fcpe` is faster and a good fit for speech. `pm` is a lightweight fallback for when the others misbehave.

**What do Envelope mix and Consonant protect do?**
Envelope mix blends the source's volume contour into the output — higher values keep the original dynamics. Consonant protect prevents unvoiced consonants (s, t, k, …) from being warped by the index or pitch shifts.

**Transpose values?**
Semitones. `+12` is one octave up, `-12` is one octave down. Pick the value that moves the source into the target voice's comfortable range.

## Separation

**What does the Separate page do?**
It splits audio into two stems with the bundled pymss models: vocals/accompaniment, dereverb, or main-vocal extraction. The "aggressive" variants cut deeper but may cost quality.

## Training

**Recommended workflow?**
1. Training page — set the experiment name, sample rate and version.
2. Step 2a — slice and normalize a dataset folder or a dropped `.zip`.
3. Step 2b — extract HuBERT and pitch features.
4. Step 3 — train the model, then train the index.

**How much data do I need?**
10–30 minutes of clean, single-speaker audio is a solid start. More helps, but noisy recordings hurt more than a smaller clean dataset.

**Multi-speaker training?**
Pick the multiple-speakers dataset type and either name subfolders `Name_ID_Repeat` (IDs 0–109) or submit a manifest on the Speakers page.

## Library and checkpoints

**What can I import?**
Model archives (`.zip`) from your device or via a link (Drive, Mega, Yandex, MediaFire, Hugging Face), and pretrained v2 G/D pairs for fine-tuning.

**What are the checkpoint tools for?**
Merge two models with a weight, edit embedded model info, inspect a checkpoint's metadata, and extract a small inference-only model from a training checkpoint.

## Troubleshooting

**The device readout shows CPU.**
CUDA wasn't detected. Install a CUDA-enabled PyTorch build from the matching `requirments_cu*.txt` file.

**Conversion fails with an index error.**
The selected `.index` doesn't match the model or is missing. Pick **Do not use index** or retrain the index for that voice.