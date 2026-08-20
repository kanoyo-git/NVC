# NVC 命令行训练与离线推理

[English documentation](../en/cli.md)

本文档只介绍 NVC 的训练和离线音色转换。PyMSS/UVR5 音源分离使用另一套 CLI。

## 运行目录

所有命令均在实际安装的项目根目录执行。将下面的占位内容替换为自己的路径：

```powershell
Set-Location "<NVC项目根目录>"
```

选择项目内置 Python 或已经安装好项目依赖的系统 Python。只保留其中一行：

```powershell
$PYTHON = "runtime\python.exe"  # 项目内置 Python
# $PYTHON = "python"            # 系统 Python
& $PYTHON --version
```

## 单说话人与多说话人的关系

模型训练核心始终包含说话人 ID 嵌入，因此从模型结构角度看，多说话人训练覆盖单说话人场景。只含一个 ID 的多说话人清单也可以训练。

当前项目仍然保留两套明确的数据约定：

| 项目 | 单说话人 | 多说话人 |
| --- | --- | --- |
| 数据入口 | 目录第一层的音频文件 | 根目录下的 `名称_ID_重复次数` 子目录 |
| 说话人 ID | 训练时统一指定一个 ID | 每条音频由 manifest 指定 ID |
| 小模型元数据 | 不写 `speaker_info` | 写入名称和 ID 的 `speaker_info` |
| 推理控件 | 默认 ID 0，可显式指定其他有效 ID | 显示名称和 ID，默认最小 ID |
| 总特征文件 | `total_fea.npy` | 每人一份 `total_fea_spkid<ID>.npy` |
| added 索引 | 普通索引文件名 | 每人一份 `_spkid<ID>.index` |

只有一个音色时建议使用单说话人模式。需要在同一模型中按名称切换音色时使用多说话人模式。

## 训练流程

训练阶段依次为：数据切分、F0 提取、HuBERT 特征提取、模型训练、索引训练。

### 单说话人数据切分

```powershell
& $PYTHON train\preprocess.py "训练集目录" 40000 8 "logs\实验名" False 3.7
```

参数依次为输入目录、采样率、进程数、实验目录、是否关闭多进程、切分参数。

### 多说话人 manifest 与数据切分

子目录格式为 `名称_ID_重复次数`，ID 范围为 `0~109`。

```powershell
& $PYTHON -c "from tools.multispeaker import build_manifest_from_root,write_manifest; write_manifest(r'logs\实验名',build_manifest_from_root(r'训练集总目录'))"

& $PYTHON train\preprocess.py "" 40000 8 "logs\实验名" False 3.7 "logs\实验名\multispeaker_manifest.json"
```

重复次数只在最终 `filelist.txt` 中重复训练记录，不会重复执行切分、F0 或 HuBERT 特征提取。

### F0 提取

CPU + PM：

```powershell
& $PYTHON train\dataset\extract_f0.py cpu "logs\实验名" 8 pm
```

CPU + RMVPE：

```powershell
& $PYTHON train\dataset\extract_f0.py cpu "logs\实验名" 8 rmvpe
```

CUDA + RMVPE：

```powershell
& $PYTHON train\dataset\extract_f0.py cuda 1 0 0 "logs\实验名" true
```

CUDA 参数依次为总进程数、当前进程序号、GPU ID、实验目录和半精度开关。多个进程需要分别启动不同的当前进程序号。

DirectML + RMVPE：

```powershell
& $PYTHON train\dataset\extract_f0.py dml "logs\实验名"
```

### HuBERT 特征提取

CUDA：

```powershell
& $PYTHON train\dataset\extract_hubert_feature.py cuda:0 1 0 0 "logs\实验名" v2 true
```

CPU：

```powershell
& $PYTHON train\dataset\extract_hubert_feature.py cpu 1 0 "logs\实验名" v2 false
```

DirectML：

```powershell
& $PYTHON train\dataset\extract_hubert_feature.py privateuseone:0 1 0 "logs\实验名" v2 false
```

### 准备 `config.json` 和 `filelist.txt`

`train.py` 启动前必须存在：

```text
logs\实验名\config.json
logs\实验名\filelist.txt
```

单说话人带 F0 的每行格式：

```text
wav路径|HuBERT特征路径|粗F0路径|连续F0路径|说话人ID
```

多说话人带 F0 的每行格式：

```text
wav路径|HuBERT特征路径|粗F0路径|连续F0路径|说话人ID|说话人名称
```

无 F0 时分别去掉两个 F0 路径字段。多说话人的 `config.json` 还需要：

```json
{
  "model": {
    "spk_embed_dim": 110
  },
  "speaker_info": [
    {"id": 0, "name": "说话人A"},
    {"id": 1, "name": "说话人B"}
  ]
}
```

GUI 的“训练模型”和“一键训练”会自动生成这两个文件；纯 CLI 流程需要在执行 `train.py` 前准备好它们。

### 模型训练

```powershell
& $PYTHON train\train.py -e "实验名" -sr 40k -f0 1 -bs 8 -g 0 -te 200 -se 5 -pg assets\pretrained_v2\f0G40k.pth -pd assets\pretrained_v2\f0D40k.pth -l 0 -c 0 -sw 1 -v v2
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `-e` | 实验名，对应 `logs\实验名` |
| `-sr` | `32k`、`40k` 或 `48k` |
| `-f0` | `1` 使用 F0，`0` 不使用 |
| `-bs` | 每张卡的 batch size |
| `-g` | GPU ID，例如 `0` 或 `0-1`；CPU 训练时可省略 |
| `-te` | 总 epoch，最大 1200 |
| `-se` | 保存间隔 epoch |
| `-l` | `1` 只保留最新大 checkpoint |
| `-c` | `1` 缓存训练集到显存 |
| `-sw` | `1` 同时保存推理小模型 |
| `-v` | `v1` 或 `v2` |

### 索引训练

单说话人：

```powershell
& $PYTHON train\train_index.py "实验名" v2 "assets\indices" 8 single
```

多说话人：

```powershell
& $PYTHON train\train_index.py "实验名" v2 "assets\indices" 8 multi
```

多说话人模式不是把所有人的特征混成一个索引。脚本会按说话人 ID 分组，为每个说话人分别生成 `total_fea_spkid<ID>.npy`、`trained_..._spkid<ID>.index` 和 `added_..._spkid<ID>.index`。因此模型中每个说话人都有自己的一份索引；推理选择说话人后，应使用该 ID 对应的 added 索引。

自动按 manifest 判断：

```powershell
& $PYTHON train\train_index.py "实验名" v2 "assets\indices" 8 auto
```

## 离线推理 CLI

入口为 `infer/cli.py`。

查看帮助：

```powershell
& $PYTHON infer\cli.py --help
```

### 查看模型的说话人

```powershell
& $PYTHON infer\cli.py --model "assets\weights\模型.pth" --list-speakers
```

带 `speaker_info` 的多说话人模型会输出 ID 和名称。旧模型或单说话人模型会输出可用 ID 范围。

### 单说话人模型推理

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\单说话人.pth" `
  --input "D:\input.wav" `
  --output "D:\output.wav" `
  --pitch 0 `
  --f0-method rmvpe `
  --index-rate 0.75
```

单说话人默认使用 ID 0。旧模型含多个未命名 ID 时可通过 `--speaker-id` 指定。

### 多说话人模型推理

省略 `--speaker-id` 时使用 checkpoint 中最小的说话人 ID，并自动匹配该 ID 的 `_spkid<ID>.index`：

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\多说话人.pth" `
  --input "D:\input.wav" `
  --output "D:\output.wav" `
  --index-rate 1
```

指定说话人：

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\多说话人.pth" `
  --speaker-id 1 `
  --input "D:\input.wav" `
  --output "D:\output.wav" `
  --index-rate 1
```

### 批量推理

目录输入只扫描当前层；加入 `--recursive` 后递归扫描并保留相对目录：

```powershell
& $PYTHON infer\cli.py `
  --model "assets\weights\模型.pth" `
  --speaker-id 0 `
  --input "D:\input_audio" `
  --output "D:\output_audio" `
  --format flac `
  --recursive `
  --overwrite
```

支持的输出格式为 `wav`、`flac`、`mp3` 和 `m4a`。输入扩展名判断不区分大小写。

### 索引规则

- 省略 `--index`：按模型名和说话人 ID 自动匹配 added 索引。
- 指定 `--index`：使用该路径；trained 文件名会转换为对应 added 文件名。
- `--index-rate` 大于 0 且索引不存在：命令以错误退出，不执行静默降级。
- 不使用索引：传入 `--index-rate 0`。

常用推理参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--speaker-id` | 单说话人 0；多说话人最小 ID | 推理说话人 |
| `--pitch` | `0` | 半音变调 |
| `--f0-method` | `rmvpe` | `pm` 或 `rmvpe` |
| `--autotune` / `--auto-register` | 关闭 | 启用基于数据集配置的新版动态 Autotune |
| `--fallback-pitch-hz` | `155` | 模型无音高配置时的目标中值 |
| `--index-rate` | `0.75` | 索引检索占比 |
| `--resample-sr` | `0` | 输出重采样率，0 表示模型采样率 |
| `--rms-mix-rate` | `1.0` | 输出音量包络混合比例 |
| `--protect` | `0.33` | 清辅音和呼吸声保护，范围 0~0.5 |
| `--overwrite` | 关闭 | 覆盖已有输出 |

正常完成返回退出码 0；发生推理失败返回 1；用户中断返回 130。

## GUI 当前 PyMSS 模型推理 CLI

这里只列出 GUI“人声伴奏分离&去混响”页面实际提供的 5 个模型。PyMSS 必须以模块方式调用：

```powershell
& $PYTHON -m tools.pymss.cli infer --help
```

GUI 名称与 CLI 模型名对应如下：

| GUI 处理方式 | CLI 模型名 |
| --- | --- |
| 去混响 | `dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` |
| 去混响（激进） | `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` |
| 去伴奏 | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` |
| 去伴奏（激进） | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| 提主旋律 | `model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` |

`--input` 可以是单个音频文件或文件夹，`--output` 必须是输出文件夹。模型文件在 CLI 缓存中不存在时会自动下载。

### 去混响

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt" `
  --input "<输入音频或目录>" `
  --output "<输出目录>" `
  --device auto `
  --format flac
```

### 去混响（激进）

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt" `
  --input "<输入音频或目录>" `
  --output "<输出目录>" `
  --device auto `
  --format flac
```

### 去伴奏

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "model_bs_roformer_ep_368_sdr_12.9628.ckpt" `
  --input "<输入音频或目录>" `
  --output "<输出目录>" `
  --device auto `
  --format flac
```

### 去伴奏（激进）

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "model_bs_roformer_ep_317_sdr_12.9755.ckpt" `
  --input "<输入音频或目录>" `
  --output "<输出目录>" `
  --device auto `
  --format flac
```

### 提主旋律

```powershell
& $PYTHON -m tools.pymss.cli infer `
  "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt" `
  --input "<输入音频或目录>" `
  --output "<输出目录>" `
  --device auto `
  --format flac
```

目录批处理时加入 `--save-as-folder`，可为每个输入音频建立单独的结果子目录。CUDA 可以将 `--device auto` 换成 `--device cuda --device-id 0`；CPU 可以换成 `--device cpu`。输出格式支持 `wav`、`flac`、`mp3` 和 `m4a`。
