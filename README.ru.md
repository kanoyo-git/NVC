<div align="center">

[English](./README.md) | [Русский](./README.ru.md)

# NVC

### Инструментарий для retrieval-based конвертации голоса, обучения моделей и работы с вокалом

[![Лицензия: MIT](https://img.shields.io/badge/license-MIT-2f6f4e.svg)](./LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-356f9f.svg)](https://www.python.org/downloads/)
[![Интерфейс: Studio](https://img.shields.io/badge/interface-NVC%20Studio-6f4eb5.svg)](#запуск-nvc)

[![Colab: English](https://img.shields.io/badge/Colab-English-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kanoyo-git/NVC/blob/main/NVC.ipynb)
[![Colab: Русский](https://img.shields.io/badge/Colab-%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kanoyo-git/NVC/blob/main/NVC.ru.ipynb)

[О проекте](#что-такое-nvc) · [Возможности](#возможности) · [Быстрый старт](#быстрый-старт) · [Динамический автотюн](#динамический-автотюн-по-профилю-датасета) · [Благодарности](#использованные-проекты-и-благодарности)

</div>

## Что такое NVC?

NVC — локальный фреймворк конвертации голоса, основанный на подходе RVC. Он преобразует тембр исходного голоса в голос обученной целевой модели, сохраняя фразировку, ритм и артикуляцию оригинального исполнения.

Система объединяет извлечение высоты тона, нейросетевой синтез голоса и поиск признаков в обучающем материале целевой модели. Retrieval-механизм уменьшает проникновение исходного тембра и приближает результат к выбранному голосу. В NVC также входят обучение моделей, пакетный инференс, разделение вокала и инструментала, конвертация голоса в реальном времени и динамический автотюн для вокала с учётом датасета.

NVC работает локально. Записи и модели остаются на вашем компьютере, если вы сами не решите загрузить или опубликовать их где-либо ещё.

## Возможности

| Раздел | Что предоставляет NVC |
| --- | --- |
| Конвертация голоса | Обработка отдельных файлов и папок с опциональным retrieval-индексом |
| Работа с высотой тона | Извлечение F0 через RMVPE, FCPE и PM, ручное транспонирование и динамический автотюн по профилю датасета |
| Обучение моделей | Одноголосые и многоголосые модели, препроцессинг, извлечение признаков и инструменты контрольных точек |
| Интерфейсы | Современный NVC Studio, классический Gradio GUI, офлайн CLI и конвертер голоса в реальном времени |
| Работа с вокалом | Разделение вокала и инструментала через модели pymss/MSST |
| Инструменты моделей | Работа с индексами, извлечение и объединение контрольных точек |
| Оборудование | Ускорение NVIDIA CUDA; CPU-режим для AMD и Intel, поддержка DirectML в Windows |

Рекомендация исходного RVC по датасету остаётся актуальной: используйте не менее 10 минут чистой речи или изолированного вокала с низким уровнем шума. Более разнообразный и качественно записанный материал обычно даёт более устойчивую модель.

## Быстрый старт

Эта ветка рассчитана на **64-битный Python 3.12**. Выполняйте команды из корня репозитория. Рекомендуемая среда Linux — Ubuntu 24.04 x86_64.

### 1. Клонирование репозитория

```bash
git clone https://github.com/kanoyo-git/NVC.git
cd NVC
```

### 2. Создание виртуального окружения

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

### 3. Установка зависимостей для вашего оборудования

CPU, AMD или Intel:

```bash
python -m pip install -r requirments_cpu_py312.txt
```

NVIDIA RTX 50-й серии:

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

Видеокарты NVIDIA до RTX 50-й серии:

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

Проверка установки:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

В файлах зависимостей по умолчанию указаны зеркала для материкового Китая. При необходимости замените только значения `--index-url` и `--extra-index-url` на официальные индексы PyPI и PyTorch. Не меняйте версии пакетов, суффиксы CUDA и порядок двухэтапной установки.

### 4. Загрузка системных моделей

Необходимые исходные ресурсы размещены в [репозитории моделей VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main).

```bash
python -m pip install --upgrade huggingface_hub

# Для инференса и извлечения признаков
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe

# Для обучения моделей v1/v2
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pretrained/*" "pretrained_v2/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
python -m zipfile -e .model-downloads/mute.zip logs

# Только для разделения вокала через pymss/MSST
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pymss_weights/*" --local-dir assets
```

Для DirectML на AMD/Intel в Windows дополнительно требуется:

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

Если FFmpeg не установлен в Windows системно, положите [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true) и [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true) в корень репозитория.

## Модели и каталоги

NVC создаёт рабочие каталоги автоматически. Пользовательские модели и индексы должны находиться в следующих папках:

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
├── weights/          # пользовательские модели голоса .pth
└── indices/          # пользовательские файлы .index
logs/
└── mute/             # образцы тишины для обучения
```

## Запуск NVC

Запуск современного NVC Studio:

```bash
python gui.py
```

Интерфейс открывается в браузере и по умолчанию использует порт `7865`. Для сервера без рабочего стола:

```bash
python gui.py --noautoopen
```

Запуск классического интерфейса Gradio:

```bash
python gui.py --legacy
```

Запуск конвертера голоса в реальном времени:

```bash
python realtime_gui.py
```

Для Windows также предусмотрены файлы `go-gui.bat` и `go-realtime_gui.bat`.

### Офлайн CLI

```bash
python -m infer.cli \
  --model assets/weights/voice.pth \
  --input input.flac \
  --output output.flac \
  --f0-method rmvpe
```

Команда `python -m infer.cli --help` покажет параметры пакетной конвертации, рекурсивного обхода каталогов, выбора исполнителя и формата результата.

## Динамический автотюн по профилю датасета

Динамический автотюн адаптирует конвертацию под вокальный диапазон целевой модели и не транспонирует всю песню вслепую. Он анализирует датасет модели, сохраняет переиспользуемый профиль высоты тона рядом с моделью и обрабатывает исходный вокал по фразам. Решения о смене октавы сглаживаются во времени, чтобы не рвать протяжённые ноты, не уничтожать вибрато и не менять тайминг песни.

В NVC Studio или классическом GUI:

1. Выберите модель голоса.
2. Включите **Динамический автотюн**.
3. Выберите папку датасета или укажите один репрезентативный аудиофайл.
4. Создайте профиль, затем обработайте вокал.

Созданный файл называется `MODEL.pitch.json` и автоматически используется при следующих конвертациях.

Создание профиля и конвертация одной командой CLI:

```bash
python -m infer.cli \
  --model assets/weights/voice.pth \
  --input song.flac \
  --output converted.flac \
  --pitch-profile-dataset /path/to/voice-dataset \
  --autotune
```

Профиль также можно создать отдельно:

```bash
python -m tools.build_pitch_profile \
  /path/to/voice-dataset \
  --model assets/weights/voice.pth
```

## Использованные проекты и благодарности

NVC основан на работе [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) и его участников. Архитектура retrieval-based конвертации голоса, процесс обучения и значительная часть первоначальной основы приложения были созданы этим проектом.

Унаследованные базовые модели RVC были обучены почти на 50 часах качественного аудио из открытого датасета VCTK. NVC сохраняет благодарность авторов исходного проекта создателям и участникам этого датасета.

Основные проекты и исследования, которые используются в NVC или были унаследованы от RVC:

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

Предобученная модель RMVPE была обучена и протестирована [yxlllc](https://github.com/yxlllc/RMVPE) и [RVC-Boss](https://github.com/RVC-Boss), как указано в благодарностях исходного проекта.

Спасибо всем, кто сообщал об ошибках, присылал исправления, тестировал модели, переводил документацию, создавал исследования и участвовал в разработке:

- [Участники NVC](https://github.com/kanoyo-git/NVC/graphs/contributors)
- [Участники RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors)

## Лицензия и ответственное использование

NVC распространяется по [лицензии MIT](./LICENSE). Сторонние компоненты и модели могут иметь собственные лицензии; ознакомьтесь с ними перед распространением или коммерческим использованием.

Используйте только те записи, датасеты и модели голосов, на которые у вас есть необходимые права и согласие. Ответственность за созданное и распространённое с помощью программы аудио несёт пользователь.
