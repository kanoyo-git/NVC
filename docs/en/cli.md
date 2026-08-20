# NVC Command-Line Training and Offline Inference

[中文文档](../cn/cli.md)

This document covers NVC training and offline voice conversion. PyMSS/UVR5 source separation uses a separate CLI.

## Working directory

Run every command from the project's actual installation root. Replace the placeholder with your own path:

```powershell
Set-Location "<NVC_PROJECT_ROOT>"
```

Select either the bundled Python runtime or a system Python that has the project dependencies installed. Keep one assignment:

```powershell
$PYTHON = "runtime\python.exe"  # Bundled Python
# $PYTHON = "python"            # System Python
& $PYTHON --version
```

## Single-speaker and multi-speaker relationship

The training architecture always includes a speaker-ID embedding. At the model level, multi-speaker training therefore includes the single-speaker case, and a multi-speaker manifest containing only one ID is valid.

The project still maintains two explicit data conventions:

| Area | Single-speaker | Multi-speaker |
| --- | --- | --- |
| Dataset input | Audio files directly inside one directory | `Name_ID_Repeat` subdirectories below a root directory |
| Speaker ID | One ID assigned to every training record | One ID per manifest entry |
| Small-model metadata | No `speaker_info` key | Named IDs stored in `speaker_info` |
| Inference selection | ID 0 by default; another valid ID may be explicit | Names and IDs are exposed; the smallest ID is the default |
| Total feature file | `total_fea.npy` | One `total_fea_spkid<ID>.npy` per speaker |
| Added index | Standard index filename | One `_spkid<ID>.index` per speaker |

Use single-speaker mode for one voice. Use multi-speaker mode when several named voices must be selectable from one model.

## Training workflow

The stages are dataset preprocessing, F0 extraction, HuBERT feature extraction, model training, and index training.

### Single-speaker preprocessing

```powershell
& $PYTHON train\preprocess.py "DATASET_DIRECTORY" 40000 8 "logs\EXPERIMENT" False 3.7
```

The arguments are input directory, sample rate, worker count, experiment directory, disable-multiprocessing flag, and slicing parameter.

### Multi-speaker manifest and preprocessing

Subdirectory names use `Name_ID_Repeat`, and IDs range from `0` through `109`.

```powershell
& $PYTHON -c "from tools.multispeaker import build_manifest_from_root,write_manifest; write_manifest(r'logs\EXPERIMENT',build_manifest_from_root(r'DATASET_ROOT'))"

& $PYTHON train\preprocess.py "" 40000 8 "logs\EXPERIMENT" False 3.7 "logs\EXPERIMENT\multispeaker_manifest.json"
```

The repeat count duplicates final `filelist.txt` records only. Preprocessing, F0 extraction, and HuBERT extraction are not repeated.

### F0 extraction

CPU with PM:

```powershell
& $PYTHON train\dataset\extract_f0.py cpu "logs\EXPERIMENT" 8 pm
```

CPU with RMVPE:

```powershell
& $PYTHON train\dataset\extract_f0.py cpu "logs\EXPERIMENT" 8 rmvpe
```

CUDA with RMVPE:

```powershell
& $PYTHON train\dataset\extract_f0.py cuda 1 0 0 "logs\EXPERIMENT" true
```

The CUDA arguments are total process count, current process index, GPU ID, experiment directory, and half-precision flag. Start one command for each process index when using several processes.

DirectML with RMVPE:

```powershell
& $PYTHON train\dataset\extract_f0.py dml "logs\EXPERIMENT"
```

### HuBERT feature extraction

CUDA:

```powershell
& $PYTHON train\dataset\extract_hubert_feature.py cuda:0 1 0 0 "logs\EXPERIMENT" v2 true
```

CPU:

```powershell
& $PYTHON train\dataset\extract_hubert_feature.py cpu 1 0 "logs\EXPERIMENT" v2 false
```

DirectML:

```powershell
& $PYTHON train\dataset\extract_hubert_feature.py privateuseone:0 1 0 "logs\EXPERIMENT" v2 false
```

### Prepare `config.json` and `filelist.txt`

Before starting `train.py`, these files must exist:

```text
logs\EXPERIMENT\config.json
logs\EXPERIMENT\filelist.txt
```

A single-speaker F0 record has this format:

```text
wav_path|HuBERT_feature_path|coarse_F0_path|continuous_F0_path|speaker_ID
```

A multi-speaker F0 record adds the speaker name:

```text
wav_path|HuBERT_feature_path|coarse_F0_path|continuous_F0_path|speaker_ID|speaker_name
```

Remove both F0 path fields for a non-F0 model. A multi-speaker `config.json` also needs:

```json
{
  "model": {
    "spk_embed_dim": 110
  },
  "speaker_info": [
    {"id": 0, "name": "Speaker A"},
    {"id": 1, "name": "Speaker B"}
  ]
}
```

The GUI Model Training and One-click Training actions generate both files automatically. A pure CLI workflow prepares them before invoking `train.py`.

### Model training

```powershell
& $PYTHON train\train.py -e "EXPERIMENT" -sr 40k -f0 1 -bs 8 -g 0 -te 200 -se 5 -pg assets\pretrained_v2\f0G40k.pth -pd assets\pretrained_v2\f0D40k.pth -l 0 -c 0 -sw 1 -v v2
```

Common options:

| Option | Meaning |
| --- | --- |
| `-e` | Experiment name under `logs` |
| `-sr` | `32k`, `40k`, or `48k` |
| `-f0` | `1` enables F0; `0` disables it |
| `-bs` | Batch size per GPU |
| `-g` | GPU IDs such as `0` or `0-1`; omit for CPU training |
| `-te` | Total epochs, up to 1200 |
| `-se` | Checkpoint interval in epochs |
| `-l` | `1` keeps only the latest large checkpoint |
| `-c` | `1` caches the dataset in GPU memory |
| `-sw` | `1` also saves inference-ready small models |
| `-v` | `v1` or `v2` |

### Index training

Single-speaker:

```powershell
& $PYTHON train\train_index.py "EXPERIMENT" v2 "assets\indices" 8 single
```

Multi-speaker:

```powershell
& $PYTHON train\train_index.py "EXPERIMENT" v2 "assets\indices" 8 multi
```

Multi-speaker mode does not merge every speaker's features into one index. The script groups features by speaker ID and separately creates `total_fea_spkid<ID>.npy`, `trained_..._spkid<ID>.index`, and `added_..._spkid<ID>.index` for each speaker. Every speaker in the model therefore has its own index; after selecting a speaker for inference, use the added index for that ID.

Automatic manifest detection:

```powershell
& $PYTHON train\train_index.py "EXPERIMENT" v2 "assets\indices" 8 auto
```

## Offline inference CLI

The entry point is `infer/cli.py`.

Show help:

```powershell
& $PYTHON infer\cli.py --help
```

### List model speakers

```powershell
& $PYTHON infer\cli.py --model "assets\weights\MODEL.pth" --list-speakers
```

Multi-speaker models with `speaker_info` print IDs and names. Older or single-speaker models print the valid ID range.

### Single-speaker inference

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\SINGLE.pth" `
  --input "D:\input.wav" `
  --output "D:\output.wav" `
  --pitch 0 `
  --f0-method rmvpe `
  --index-rate 0.75
```

Single-speaker models default to ID 0. Use `--speaker-id` for an older model that contains several unnamed IDs.

### Dataset-aware dynamic autotune

The new autotune corrects note centres in log-frequency space while retaining vibrato and pitch glides, then compares each complete vocal phrase with the model profile. It may change octaves only in unvoiced gaps, so it does not change the song key, split a held note, or alter timing:

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\SINGER.pth" `
  --input "D:\vocal.wav" `
  --output "D:\converted.wav" `
  --pitch-profile-dataset "D:\dataset\Singer"
```

`--pitch-profile-dataset` accepts either one audio file or a directory (scanned recursively), creates `MODEL.pitch.json`, and enables dynamic register matching. On later runs, omit it and use `--auto-register` to reuse the saved profile.

Build a sidecar from a single file or recursively from the vocal dataset used to train a model. `--model` automatically writes `MODEL.pitch.json` beside the checkpoint:

```powershell
& $PYTHON tools\build_pitch_profile.py "D:\dataset\Singer" `
  --model "assets\weights\SINGER.pth"
```

If a folder contains several voices, use the stem-specific name `MODEL.pitch.json`. It takes precedence over the folder-level profile. Without a profile, `--fallback-pitch-hz` is used as a median target. The old `--proposed-pitch` and `--proposed-pitch-threshold` spellings remain as CLI aliases.

### Multi-speaker inference

When `--speaker-id` is omitted, the CLI selects the smallest declared ID and automatically matches its `_spkid<ID>.index`:

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\MULTI.pth" `
  --input "D:\input.wav" `
  --output "D:\output.wav" `
  --index-rate 1
```

Select a specific speaker:

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\MULTI.pth" `
  --speaker-id 1 `
  --input "D:\input.wav" `
  --output "D:\output.wav" `
  --index-rate 1
```

### Batch inference

Directory input scans direct child files. Add `--recursive` to scan subdirectories and preserve their relative layout:

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\MODEL.pth" `
  --speaker-id 0 `
  --input "D:\input_audio" `
  --output "D:\output_audio" `
  --format flac `
  --recursive `
  --overwrite
```

Output formats are `wav`, `flac`, `mp3`, and `m4a`. Input extension matching is case-insensitive.

### Index behavior

- Omit `--index` to match an added index by model name and speaker ID.
- Supply `--index` to select a path explicitly. A trained filename is converted to its added counterpart.
- If `--index-rate` is greater than zero and the index is missing, the command exits with an error instead of silently disabling retrieval.
- Use `--index-rate 0` to disable retrieval.

Common inference options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--speaker-id` | Single: 0; multi: smallest declared ID | Inference speaker |
| `--pitch` | `0` | Pitch shift in semitones |
| `--f0-method` | `rmvpe` | `pm` or `rmvpe` |
| `--autotune` / `--auto-register` | Off | Run the new dataset-aware dynamic autotune |
| `--pitch-profile-dataset` | None | Build the model profile from one audio file or a recursive dataset folder |
| `--fallback-pitch-hz` | `155` | Target median if no profile is available |
| `--index-rate` | `0.75` | Retrieval blend ratio |
| `--resample-sr` | `0` | Output sample rate; 0 keeps the model rate |
| `--rms-mix-rate` | `1.0` | Output RMS-envelope blend ratio |
| `--protect` | `0.33` | Consonant and breath protection, from 0 to 0.5 |
| `--overwrite` | Off | Replace existing output files |

Successful completion returns exit code 0, an inference failure returns 1, and user interruption returns 130.

## CLI Inference For The Current GUI PyMSS Models

This section only covers the five models exposed by the GUI's Vocal/Instrumental Separation and Dereverb page. Invoke PyMSS as a module:

```powershell
& $PYTHON -m tools.pymss.cli infer --help
```

The GUI labels map to CLI catalog model names as follows:

| GUI action | CLI model name |
| --- | --- |
| Dereverb | `dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` |
| Dereverb (aggressive) | `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` |
| Remove instrumental | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` |
| Remove instrumental (aggressive) | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| Extract lead vocal | `model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` |

`--input` accepts one audio file or a directory, while `--output` is an output directory. Missing files in the CLI model cache are downloaded automatically.

### Dereverb

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt" `
  --input "<input audio or directory>" `
  --output "<output directory>" `
  --device auto `
  --format flac
```

### Dereverb (Aggressive)

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt" `
  --input "<input audio or directory>" `
  --output "<output directory>" `
  --device auto `
  --format flac
```

### Remove Instrumental

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "model_bs_roformer_ep_368_sdr_12.9628.ckpt" `
  --input "<input audio or directory>" `
  --output "<output directory>" `
  --device auto `
  --format flac
```

### Remove Instrumental (Aggressive)

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "model_bs_roformer_ep_317_sdr_12.9755.ckpt" `
  --input "<input audio or directory>" `
  --output "<output directory>" `
  --device auto `
  --format flac
```

### Extract Lead Vocal

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt" `
  --input "<input audio or directory>" `
  --output "<output directory>" `
  --device auto `
  --format flac
```

For directory batches, add `--save-as-folder` to create a separate result folder for each input. On CUDA, replace `--device auto` with `--device cuda --device-id 0`; for CPU, use `--device cpu`. Output formats are `wav`, `flac`, `mp3`, and `m4a`.
