<div align="center">

<h1>Retrieval-based-Voice-Conversion-GUI</h1>
Um framework simples e fácil de usar para conversão de timbre vocal / alteração de voz.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/NVC-Project/Retrieval-based-Voice-Conversion-GUI)

<img src="https://counter.seku.su/cmoe?name=nvc&theme=r34" /><br>

[![Licence](https://img.shields.io/github/license/NVC-Project/Retrieval-based-Voice-Conversion-GUI?style=for-the-badge)](https://github.com/NVC-Project/Retrieval-based-Voice-Conversion-GUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionGUI/tree/main/)


</div>

------
[**Changelog**](./Changelog_pt.md) | [**FAQ (Frequently Asked Questions)**](https://github.com/NVC-Project/Retrieval-based-Voice-Conversion-GUI/wiki/FAQ-(Frequently-Asked-Questions))

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)


Confira nosso [Vídeo de demonstração](https://www.bilibili.com/video/BV1pm4y1z7Gm/) aqui!

Treinamento/Inferência GUI：go-gui.bat
![Traduzido](https://github.com/RafaelGodoyEbert/Retrieval-based-Voice-Conversion-GUI/assets/78083427/0b894d87-565a-432c-8b5b-45e4a65d5d17)

GUI de conversão de voz em tempo real：go-realtime_gui.bat
![image](https://github.com/RafaelGodoyEbert/Retrieval-based-Voice-Conversion-GUI/assets/78083427/d172e3e5-35f4-4876-9530-c28246919e9e)


> O dataset para o modelo de pré-treinamento usa quase 50 horas de conjunto de dados de código aberto VCTK de alta qualidade.

> Dataset de músicas licenciadas de alta qualidade serão adicionados ao conjunto de treinamento, um após o outro, para seu uso, sem se preocupar com violação de direitos autorais.

> Aguarde o modelo básico pré-treinado do NVCv3, que possui parâmetros maiores, mais dados de treinamento, melhores resultados, velocidade de inferência inalterada e requer menos dados de treinamento para treinamento.

## Resumo
Este repositório possui os seguintes recursos:
+ Reduza o vazamento de tom substituindo o recurso de origem pelo recurso de conjunto de treinamento usando a recuperação top1;
+ Treinamento fácil e rápido, mesmo em placas gráficas relativamente ruins;
+ Treinar com uma pequena quantidade de dados também obtém resultados relativamente bons (>=10min de áudio com baixo ruído recomendado);
+ Suporta fusão de modelos para alterar timbres (usando guia de processamento ckpt-> mesclagem ckpt);
+ Interface GUI fácil de usar;
+ Use o modelo pymss/MSST para separar rapidamente vocais e instrumentos.
+ Use o mais poderoso algoritmo de extração de voz de alta frequência [InterSpeech2023-RMVPE](#Credits) para evitar o problema de som mudo. Fornece os melhores resultados (significativamente) e é mais rápido, com consumo de recursos ainda menor que o Crepe_full.
+ Sistemas AMD/Intel usam as dependências de CPU; Windows pode usar DirectML e Linux usa CPU.

## Preparando o ambiente

Esta branch é destinada a **Python 3.12 x64**. Execute todos os comandos na raiz do repositório. Ubuntu 24.04 x86_64 é recomendado.

### Ubuntu 24.04

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

Instale o Python 3.12 x64 e crie um ambiente virtual:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### Escolha de dependências por hardware

| Hardware | Instalação |
| --- | --- |
| CPU, AMD, Intel | Use `requirments_cpu_py312.txt`; Windows pode usar DirectML, Linux usa CPU |
| NVIDIA RTX série 50 | Instale primeiro o Torch CUDA 12.8 e depois `requirments_cu128_py312.txt` |
| NVIDIA anterior à série RTX 50 | Instale primeiro o Torch CUDA 11.8 e depois `requirments_cu118_py312.txt` |

#### CPU, AMD, Intel

```bash
python -m pip install -r requirments_cpu_py312.txt
```

#### NVIDIA RTX série 50: duas etapas

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

#### NVIDIA anterior à série RTX 50: duas etapas

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

Verifique Torch e CUDA:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```


### Fontes de pacotes

Os três arquivos `requirments_*.txt` definem suas fontes no topo. Para usar fontes oficiais, substitua somente `--index-url` e `--extra-index-url`, mantendo versões, sufixos CUDA e a ordem das duas etapas.

| Default mirror | Official source |
| --- | --- |
| `https://mirrors.pku.edu.cn/pypi/simple` | `https://pypi.org/simple` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cpu` | `https://download.pytorch.org/whl/cpu` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu118` | `https://download.pytorch.org/whl/cu118` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | `https://download.pytorch.org/whl/cu128` |

## Modelos e diretórios de execução

O GUI cria os diretórios de execução automaticamente. Baixe os modelos do [repositório de modelos Hugging Face](https://huggingface.co/lj1995/VoiceConversionGUI/tree/main) e mantenha esta estrutura:

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── rmvpe/rmvpe.pt
├── pretrained/
├── pretrained_v2/
├── pymss_weights/
├── weights/        # user NVC .pth models
└── indices/        # user .index files
logs/
└── mute/           # training silence samples

# Exact paths used by the code
assets/hubert_base/config.json
assets/hubert_base/preprocessor_config.json
assets/hubert_base/pytorch_model.bin
assets/rmvpe/rmvpe.pt
assets/pretrained/*.pth
assets/pretrained_v2/*.pth
assets/pymss_weights/*
assets/weights/*.pth
assets/indices/*.index
logs/mute/*
```

### Baixar modelos

```bash
python -m pip install --upgrade huggingface_hub

# Required for inference and feature extraction
hf download lj1995/VoiceConversionGUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionGUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe

# Required for v1/v2 training
hf download lj1995/VoiceConversionGUI --revision main \
  --include "pretrained/*" "pretrained_v2/*" --local-dir assets
hf download lj1995/VoiceConversionGUI mute.zip --revision main \
  --local-dir .model-downloads
python -m zipfile -e .model-downloads/mute.zip logs

# Required only for pymss/MSST vocal separation
hf download lj1995/VoiceConversionGUI --revision main \
  --include "pymss_weights/*" --local-dir assets
```

Ambientes Windows AMD/Intel DirectML também precisam de:

```bash
hf download lj1995/VoiceConversionGUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```


### FFmpeg

O comando Ubuntu acima instala o FFmpeg. No Windows, coloque estes arquivos na raiz do repositório:

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionGUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionGUI/resolve/main/ffprobe.exe?download=true)

## Iniciar o GUI

```bash
python gui.py
```

Servidor Ubuntu sem interface gráfica:

```bash
python gui.py --noautoopen
```

A porta padrão é `7865`. Coloque modelos `.pth` em `assets/weights/` e arquivos `.index` em `assets/indices/`.

## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [pymss-project/pymss](https://github.com/pymss-project/pymss)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [NVC-Boss](https://github.com/NVC-Boss).

## Thanks to all contributors for their efforts
<a href="https://github.com/NVC-Project/Retrieval-based-Voice-Conversion-GUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=NVC-Project/Retrieval-based-Voice-Conversion-GUI" />
</a>
