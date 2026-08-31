import os
import shutil
import html
import copy
import re
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# Offline GUI keeps the CUDA Graph implementation available, but remains
# eager by default. Set NVC_OFFLINE_CUDA_GRAPH=1 to opt in for benchmarking or
# controlled deployments.
_offline_cuda_graph = os.environ.get("NVC_OFFLINE_CUDA_GRAPH", "0") == "1"
os.environ["NVC_CUDA_GRAPH"] = "1" if _offline_cuda_graph else "0"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("no_proxy", "localhost, 127.0.0.1, ::1")
if "matplotlib_inline" in os.environ.get("MPLBACKEND", ""):
    os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("weight_pymss_root", "assets/pymss_weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")
os.environ.setdefault("nmvpe_root", "assets/nmvpe")

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
for name in os.listdir(tmp):
    if name == "jieba.cache":
        continue
    path = os.path.join(tmp, name)
    delete = (
        os.remove if os.path.isfile(path) or os.path.islink(path) else shutil.rmtree
    )
    try:
        delete(path)
    except Exception as error:
        print(str(error))

from configs.config import Config, GPU_INDEX, GPU_INFOS, GPU_MEMORY, IS_GPU
from infer.hubert import EMBEDDER_CHOICES
from infer.vc.modules import VC
from infer.vc.utils import get_index_path_from_model
from tools.pymss_gui import (
    PYMSS_MODEL_CHOICES,
    get_model_info,
    pymss_separate as _pymss_separate_core,
    stop_pymss_separation as _stop_pymss_separation_core,
)
from tools.file_io import read_text
from tools.process_utils import kill_process_tree
from tools.multispeaker import (
    ManifestError,
    build_manifest_from_root,
    build_manifest_from_rows,
    load_manifest,
    write_manifest,
)
from train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import I18nAuto
import torch, platform
import numpy as np
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import traceback
import threading
import logging
import socket
import subprocess
import time


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def find_available_port(start_port, host="0.0.0.0"):
    """Return the first bindable TCP port at or above ``start_port``."""
    if not 1 <= start_port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {start_port}.")

    for port in range(start_port, 65536):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            return port
        except OSError:
            continue

    raise OSError(
        f"No available TCP port from {start_port} through 65535; GUI was not started."
    )


runtime_dirs = (
    os.path.join(now_dir, "logs"),
    os.environ["weight_root"],
    os.environ["weight_pymss_root"],
    os.environ["index_root"],
    os.environ["outside_index_root"],
    os.environ["rmvpe_root"],
    os.environ["nmvpe_root"],
    os.path.join(now_dir, "assets", "hubert_base"),
    os.path.join(now_dir, "assets", "pretrained_v2"),
)
for runtime_dir in runtime_dirs:
    os.makedirs(runtime_dir, exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


i18n = I18nAuto()
logger.info(i18n)
print(
    i18n("当前设备：%s | 推理精度：%s") % (config.device, config.dtype),
    flush=True,
)
# GPU filtering and precision rules are shared with inference/extraction/training.
gpu_infos = list(GPU_INFOS)
gpu_indices = sorted(GPU_INDEX)
if_gpu_ok = IS_GPU
if if_gpu_ok:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = 8
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join(str(i) for i in gpu_indices)
feature_gpus = "%s-%s" % (gpus, gpus) if gpus else ""
default_training_f0_method = "rmvpe" if IS_GPU else "pm"


weight_root = os.getenv("weight_root")
weight_pymss_root = os.getenv("weight_pymss_root")
outside_index_root = os.getenv("outside_index_root")

def weight_names():
    root = pathlib.Path(weight_root)
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.pth")
        if path.is_file()
    )


def create_model_pitch_profile(model_name, dataset_path):
    """Create a sidecar profile from a user dataset and hot-reload it."""

    from infer.vc.pitch_tuning import load_pitch_profile
    from tools.build_pitch_profile import build_pitch_profile, save_pitch_profile

    model_name = str(model_name or "").strip()
    dataset_path = str(dataset_path or "").strip()
    if not model_name:
        raise ValueError(i18n("请先选择推理音色"))
    if not dataset_path:
        raise ValueError(i18n("请输入用于音高配置的数据集路径"))
    root = pathlib.Path(weight_root).expanduser().resolve()
    model_path = (root / model_name).resolve()
    if root not in model_path.parents or not model_path.is_file():
        raise ValueError(i18n("无效的模型路径"))
    if model_path.suffix.lower() != ".pth":
        raise ValueError(i18n("模型必须是PTH文件"))

    profile = build_pitch_profile(
        dataset_path,
        name=model_path.stem,
        device=str(config.device),
    )
    output_path = save_pitch_profile(profile, model_path.with_suffix(".pitch.json"))
    loaded_profile, source = load_pitch_profile(model_path)
    if loaded_profile is None:
        raise RuntimeError(i18n("无法读取创建的音高配置"))
    if vc.model_name == model_name and vc.pipeline is not None:
        vc.pipeline.pitch_profile = loaded_profile
        vc.pipeline.pitch_profile_source = source
    return (
        output_path,
        i18n("音高配置已创建：%s 个音频，%s 个有效音高帧")
        % (profile["audio_files"], profile["voiced_frames"]),
    )


def create_model_pitch_profile_status(model_name, dataset_path):
    path, status = create_model_pitch_profile(model_name, dataset_path)
    return "%s\n%s" % (status, path)


def toggle_pitch_profile_controls(enabled):
    return {"visible": bool(enabled), "__type__": "update"}


def refresh_weight_choices(previous_names=None, force=False):
    current_names = tuple(weight_names())
    if force or current_names != previous_names:
        return current_names, change_choices()
    return current_names, {"__type__": "update"}


names = weight_names()
pymss_names = PYMSS_MODEL_CHOICES


def change_choices():
    return {"choices": weight_names(), "__type__": "update"}


def clean():
    return {"value": "", "__type__": "update"}


def component_update(**values):
    return {"__type__": "update", **values}


def selected_speaker_id(slider_value, dropdown_value):
    value = dropdown_value if dropdown_value is not None else slider_value
    if isinstance(value, str):
        match = re.search(r"ID\s*[:：]\s*(\d+)\s*[)）]?\s*$", value)
        if match:
            return int(match.group(1))
    return int(value)


def normalize_index_path(file_index):
    return (
        str(file_index or "")
        .strip(" ")
        .strip('"')
        .strip("\n")
        .strip('"')
        .strip(" ")
        .replace("trained", "added")
    )


def report_missing_index(file_index):
    index_path = normalize_index_path(file_index)
    if index_path and not os.path.isfile(index_path):
        message = i18n("索引文件不存在，将不使用索引继续推理：%s") % index_path
        print(message, flush=True)
        raise RuntimeError(message)


def update_speaker_index(model_name, slider_value, dropdown_value):
    try:
        speaker_id = selected_speaker_id(slider_value, dropdown_value)
    except (TypeError, ValueError):
        speaker_id = None
    path = get_index_path_from_model(model_name, speaker_id)
    update = {"value": path, "__type__": "update"}
    return update, dict(update)


def update_dropdown_speaker_index(model_name, dropdown_value):
    return update_speaker_index(model_name, 0, dropdown_value)


def vc_single_with_speaker(slider_value, dropdown_value, *args):
    return vc.vc_single(selected_speaker_id(slider_value, dropdown_value), *args)


def vc_multi_with_speaker(slider_value, dropdown_value, *args):
    yield from vc.vc_multi(selected_speaker_id(slider_value, dropdown_value), *args)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

MULTISPEAKER_PAGE_SIZE = 10
MULTISPEAKER_MAX_ROWS = 110


def is_multispeaker_mode(training_mode):
    return training_mode in ("多说话人", i18n("多说话人"))


def manifest_error_text(error):
    text = i18n(error.key)
    return text % error.values if error.values else text


def experiment_path(exp_name):
    return os.path.join(now_dir, "logs", str(exp_name or "").strip())


def prepare_multispeaker_manifest(trainset_dir, exp_name):
    exp_path = experiment_path(exp_name)
    try:
        if str(trainset_dir or "").strip():
            manifest = build_manifest_from_root(trainset_dir)
            write_manifest(exp_path, manifest)
            return manifest
        return load_manifest(exp_path)
    except ManifestError as error:
        raise RuntimeError(manifest_error_text(error))


def load_experiment_manifest(exp_path):
    try:
        return load_manifest(exp_path)
    except ManifestError as error:
        raise RuntimeError(manifest_error_text(error))


def change_training_mode(training_mode):
    multi = is_multispeaker_mode(training_mode)
    label = (
        i18n("多说话人训练集总文件夹路径")
        if multi
        else i18n("输入训练文件夹路径")
    )
    placeholder = i18n("留空则使用辅助页已提交的清单") if multi else ""
    textbox_update = component_update(label=label, placeholder=placeholder)
    if multi:
        textbox_update["value"] = ""
    return textbox_update, component_update(visible=not multi)


def sync_exp_name(source_value, target_value):
    if str(source_value or "") == str(target_value or ""):
        return component_update()
    return component_update(value=source_value)


def empty_multispeaker_rows():
    rows = [["", "", "", ""] for _ in range(MULTISPEAKER_MAX_ROWS)]
    rows[0] = ["", "", 0, 1]
    rows[1] = ["", "", 1, 1]
    return rows


def sync_multispeaker_page(rows, active_count, page, values):
    rows = [list(row) for row in (rows or empty_multispeaker_rows())]
    while len(rows) < MULTISPEAKER_MAX_ROWS:
        rows.append(["", "", "", ""])
    start = int(page) * MULTISPEAKER_PAGE_SIZE
    for slot in range(MULTISPEAKER_PAGE_SIZE):
        index = start + slot
        if index >= MULTISPEAKER_MAX_ROWS:
            break
        offset = slot * 4
        rows[index] = [
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]
    return rows


def multispeaker_page_updates(rows, active_count, page):
    active_count = max(2, min(MULTISPEAKER_MAX_ROWS, int(active_count)))
    total_pages = max(1, (active_count + MULTISPEAKER_PAGE_SIZE - 1) // MULTISPEAKER_PAGE_SIZE)
    page = max(0, min(int(page), total_pages - 1))
    start = page * MULTISPEAKER_PAGE_SIZE
    updates = []
    for slot in range(MULTISPEAKER_PAGE_SIZE):
        index = start + slot
        visible = index < active_count
        row = rows[index] if index < len(rows) else ["", "", "", ""]
        updates.extend(
            [
                component_update(value=row[0], visible=visible),
                component_update(value=row[1], visible=visible),
                component_update(
                    value=row[2] if row[2] not in ("", None) else None,
                    visible=visible,
                ),
                component_update(
                    value=row[3] if row[3] not in ("", None) else None,
                    visible=visible,
                ),
            ]
        )
    page_text = i18n("第%s/%s页，共%s行") % (page + 1, total_pages, active_count)
    return tuple([rows, active_count, page, page_text] + updates)


def change_multispeaker_page(action, rows, active_count, page, *values):
    rows = sync_multispeaker_page(rows, active_count, page, values)
    active_count = int(active_count)
    page = int(page)
    if action == "add":
        if active_count < MULTISPEAKER_MAX_ROWS:
            used_ids = set()
            for row in rows[:active_count]:
                try:
                    speaker_id = int(float(row[2]))
                except (TypeError, ValueError):
                    continue
                if 0 <= speaker_id <= 109:
                    used_ids.add(speaker_id)
            speaker_id = next(
                value for value in range(110) if value not in used_ids
            )
            rows[active_count] = ["", "", speaker_id, 1]
            active_count += 1
        page = (active_count - 1) // MULTISPEAKER_PAGE_SIZE
    elif action == "remove":
        if active_count > 2:
            active_count -= 1
            rows[active_count] = ["", "", "", ""]
        page = min(page, (active_count - 1) // MULTISPEAKER_PAGE_SIZE)
    elif action == "previous":
        page -= 1
    elif action == "next":
        page += 1
    return multispeaker_page_updates(rows, active_count, page)


def add_multispeaker_row(rows, active_count, page, *values):
    return change_multispeaker_page("add", rows, active_count, page, *values)


def remove_multispeaker_row(rows, active_count, page, *values):
    return change_multispeaker_page("remove", rows, active_count, page, *values)


def previous_multispeaker_page(rows, active_count, page, *values):
    return change_multispeaker_page("previous", rows, active_count, page, *values)


def next_multispeaker_page(rows, active_count, page, *values):
    return change_multispeaker_page("next", rows, active_count, page, *values)


def render_multispeaker_submit_status(message, warning=False):
    color = "#9a6700" if warning else "#116329"
    background = "#fff8c5" if warning else "#dafbe1"
    return (
        '<div class="multispeaker-submit-status" style="border-color:%s;'
        'background:%s;color:%s;">%s</div>'
    ) % (color, background, color, html.escape(message))


def submit_multispeaker_rows(exp_name, rows, active_count, page, *values):
    rows = sync_multispeaker_page(rows, active_count, page, values)
    exp_name = str(exp_name or "").strip()
    if not exp_name:
        raise RuntimeError(i18n("实验名不能为空"))
    try:
        manifest, invalid_rows = build_manifest_from_rows(rows[: int(active_count)])
        path = write_manifest(experiment_path(exp_name), manifest)
    except ManifestError as error:
        raise RuntimeError(manifest_error_text(error))
    message = i18n("多说话人训练集清单已保存：%s；有效音频%s个") % (
        path,
        len(manifest["entries"]),
    )
    if invalid_rows:
        warning_text = i18n("以下行填写不完整或无效，已忽略：%s") % ", ".join(
            str(index) for index in invalid_rows
        )
        logger.warning(warning_text)
        message = "%s\n%s" % (warning_text, message)
        return rows, render_multispeaker_submit_status(message, True)
    return rows, render_multispeaker_submit_status(message)


TRAIN_TASK_LOCK = threading.Lock()
TRAIN_TASK = None


def button_update(value=None, variant=None, visible=None):
    update = {"__type__": "update"}
    if value is not None:
        update["value"] = value
    if variant is not None:
        update["variant"] = variant
    if visible is not None:
        update["visible"] = visible
    return update


def render_pymss_progress(percent=0, label="等待开始", state="idle"):
    percent = max(0.0, min(100.0, float(percent or 0)))
    colors = {
        "idle": "#64748b",
        "running": "#2563eb",
        "done": "#15803d",
        "stopped": "#b45309",
        "failed": "#b91c1c",
    }
    color = colors.get(state, colors["running"])
    safe_label = html.escape(str(label or ""))
    return (
        '<div style="min-height:52px;padding:6px 0;">'
        '<div style="display:flex;justify-content:space-between;gap:12px;'
        'align-items:center;margin-bottom:7px;font-size:14px;line-height:20px;">'
        '<span style="overflow-wrap:anywhere;">%s</span>'
        '<strong style="flex:0 0 auto;color:%s;">%.1f%%</strong>'
        "</div>"
        '<div role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        'aria-valuenow="%.1f" style="height:10px;width:100%%;overflow:hidden;'
        'border-radius:4px;background:#e2e8f0;">'
        '<div style="height:100%%;width:%.3f%%;background:%s;"></div>'
        "</div></div>"
    ) % (safe_label, color, percent, percent, percent, color)


def run_pymss_separation(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    format0,
):
    progress_state = {
        "percent": 0.0,
        "label": "正在准备 PyMSS 分离任务",
        "state": "running",
    }
    busy = False

    def update_progress(event):
        nonlocal busy
        event_type = event.get("event")
        file_count = max(1, int(event.get("file_count") or 1))
        file_index = max(1, int(event.get("file_index") or 1))
        message = str(event.get("message") or "")

        if event_type == "progress":
            done = max(0.0, float(event.get("done") or 0))
            total = max(1.0, float(event.get("total") or 1))
            file_fraction = min(1.0, done / total)
            progress_state["percent"] = (
                (file_index - 1 + file_fraction) / file_count * 100
            )
            progress_state["label"] = "文件 %s/%s · %s · %.0f/%.0f 秒" % (
                file_index,
                file_count,
                message or "正在处理音频",
                done,
                total,
            )
            progress_state["state"] = "running"
        elif event_type == "file_start":
            progress_state["percent"] = (file_index - 1) / file_count * 100
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "file":
            progress_state["percent"] = file_index / file_count * 100
            progress_state["label"] = (
                message.splitlines()[0] if message else "文件处理结束"
            )
            progress_state["state"] = "running" if event.get("ok") else "failed"
        elif event_type == "done":
            successful = int(event.get("successful") or 0)
            failed = int(event.get("failed") or 0)
            progress_state["percent"] = 100.0
            progress_state["label"] = "分离完成：成功 %s，失败 %s" % (
                successful,
                failed,
            )
            progress_state["state"] = "done" if failed == 0 else "failed"
        elif event_type == "retry_fp32":
            progress_state["percent"] = 0.0
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "cancelled":
            progress_state["label"] = message or "PyMSS 分离任务已停止"
            progress_state["state"] = "stopped"
        elif event_type in {"fatal", "busy"}:
            progress_state["label"] = (
                message.splitlines()[0] if message else "PyMSS 分离任务失败"
            )
            progress_state["state"] = "failed"
            busy = event_type == "busy"
        elif event_type in {"preparing", "precision_attempt", "status"}:
            progress_state["label"] = message
            progress_state["state"] = "running"

    last_info = ""
    try:
        for info in _pymss_separate_core(
            model_name,
            inp_root,
            save_root_vocal,
            paths,
            save_root_ins,
            format0,
            event_callback=update_progress,
        ):
            last_info = info
            start_button = button_update()
            stop_button = button_update()
            if not busy:
                start_button = button_update(visible=False)
                stop_button = button_update(visible=True)
            yield (
                info,
                render_pymss_progress(**progress_state),
                start_button,
                stop_button,
            )
    except Exception:
        last_info = "失败\n%s" % traceback.format_exc()
        progress_state.update(
            {"label": "PyMSS 分离任务失败", "state": "failed"}
        )
        logger.exception("PyMSS GUI task failed")

    start_button = button_update()
    stop_button = button_update()
    if not busy:
        start_button = button_update(visible=True)
        stop_button = button_update(visible=False)
    yield (last_info, render_pymss_progress(**progress_state), start_button, stop_button)


def stop_pymss_gui():
    return (
        _stop_pymss_separation_core(),
        render_pymss_progress(0, "PyMSS 分离任务已停止", "stopped"),
        button_update(visible=True),
        button_update(visible=False),
    )


def format_status(title, state, detail=""):
    lines = ["【%s】" % i18n(title), "%s：%s" % (i18n("状态"), i18n(state))]
    if detail:
        lines.extend(["", detail.strip()])
    return "\n".join(lines)


def format_workflow_status(step, detail="", completed_steps=None, state="运行中"):
    completed_steps = completed_steps or []
    detail = str(detail).strip()
    lines = []
    if completed_steps:
        lines.append("%s：" % i18n("已完成阶段"))
        lines.extend(
            "✓ %s：%s" % (i18n(completed_step), i18n("已成功"))
            for completed_step in completed_steps
        )
    if step:
        if lines:
            lines.append("")
        lines.append("%s：%s" % (i18n("当前阶段"), i18n(step)))
    if detail:
        lines.extend(["", detail])
    return format_status(
        "一键训练",
        state,
        "\n".join(lines),
    )


def read_log(path, max_lines=40):
    try:
        lines = [line.rstrip() for line in read_text(path, errors="ignore").splitlines()]
        lines = [line for line in lines if line.strip()]
        if len(lines) > max_lines:
            tail_count = max(0, max_lines - 1)
            omitted = len(lines) - tail_count
            tail = lines[-tail_count:] if tail_count else []
            lines = [i18n("……已省略前%s行，仅显示最新状态") % omitted]
            lines.extend(tail)
        return "\n".join(lines)
    except FileNotFoundError:
        return ""


def artifact_names(directory, suffix):
    if not os.path.isdir(directory):
        return set()
    return {
        name.split(".")[0]
        for name in os.listdir(directory)
        if name.lower().endswith(suffix)
    }


def validate_preprocess_outputs(exp_dir):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    gt_names = artifact_names(os.path.join(exp_path, "0_gt_wavs"), ".wav")
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    if not gt_names:
        raise RuntimeError(i18n("数据切分没有生成有效训练音频，请检查训练集和数据切分日志"))
    if not wav16_names:
        raise RuntimeError(i18n("数据切分没有生成16k音频，已停止后续特征提取和训练"))
    if not gt_names & wav16_names:
        raise RuntimeError(i18n("数据切分输出文件不匹配，已停止后续特征提取和训练"))


def validate_feature_outputs(exp_dir, if_f0):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    feature_names = artifact_names(os.path.join(exp_path, "3_feature768"), ".npy")
    matched = wav16_names & feature_names
    if not feature_names or not matched:
        raise RuntimeError(i18n("HuBERT特征提取没有生成有效结果，已停止训练"))
    if if_f0:
        f0_names = artifact_names(os.path.join(exp_path, "2a_f0"), ".npy")
        f0nsf_names = artifact_names(os.path.join(exp_path, "2b-f0nsf"), ".npy")
        matched &= f0_names & f0nsf_names
        if not f0_names or not f0nsf_names or not matched:
            raise RuntimeError(i18n("F0提取没有生成有效结果，已停止训练"))
    return matched


def begin_train_task(name):
    global TRAIN_TASK
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is None:
            state = {
                "name": name,
                "processes": [],
                "stop_requested": False,
            }
            TRAIN_TASK = state
            return "start", state
        return "busy", TRAIN_TASK


def stop_train_task(name):
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is None:
            return (
                format_status(name, "未运行"),
                button_update(visible=True),
                button_update(visible=False),
            )
        if TRAIN_TASK["name"] != name:
            return (
                format_status(
                    name,
                    "无法停止",
                    i18n("%s运行中，请先停止该任务") % i18n(TRAIN_TASK["name"]),
                ),
                button_update(),
                button_update(),
            )
        state = TRAIN_TASK
        state["stop_requested"] = True
        processes = list(state["processes"])
    for process in processes:
        kill_process_tree(process, name, logger)
    return (
        format_status(name, "已停止"),
        button_update(visible=True),
        button_update(visible=False),
    )


def finish_train_task(state):
    global TRAIN_TASK
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is state:
            TRAIN_TASK = None


def train_task_stopped(state):
    with TRAIN_TASK_LOCK:
        return state["stop_requested"]


def start_train_process(state, cmd):
    kwargs = {"shell": True, "cwd": now_dir}
    if "train/train.py" in cmd.replace("\\", "/"):
        training_env = os.environ.copy()
        training_env["NVC_CUDA_GRAPH"] = "0"
        kwargs["env"] = training_env
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    logger.info("%s: %s", i18n("执行命令"), cmd)
    process = Popen(cmd, **kwargs)
    with TRAIN_TASK_LOCK:
        state["processes"].append(process)
        stopped = state["stop_requested"]
    if stopped:
        kill_process_tree(process, state["name"], logger)
    return process


def wait_train_processes(
    state,
    processes,
    log_path=None,
    title="任务",
    format_output=True,
    watch_weights=False,
):
    last_snapshot = None
    last_emit_time = 0
    last_weight_names = tuple(weight_names()) if watch_weights else ()
    while any(process.poll() is None for process in processes):
        if train_task_stopped(state):
            for process in processes:
                kill_process_tree(process, state["name"], logger)
            break
        if log_path:
            snapshot = read_log(log_path)
            current_time = time.monotonic()
            current_weight_names = tuple(weight_names()) if watch_weights else ()
            weights_changed = watch_weights and current_weight_names != last_weight_names
            if (
                snapshot != last_snapshot
                or weights_changed
                or current_time - last_emit_time >= 5
            ):
                yield (
                    format_status(title, "运行中", snapshot)
                    if format_output
                    else snapshot
                )
                last_snapshot = snapshot
                last_emit_time = current_time
                last_weight_names = current_weight_names
        sleep(1)
    with TRAIN_TASK_LOCK:
        for process in processes:
            if process in state["processes"]:
                state["processes"].remove(process)
    if log_path:
        final_state = "已停止" if train_task_stopped(state) else "正在收尾"
        snapshot = read_log(log_path)
        yield (
            format_status(title, final_state, snapshot)
            if format_output
            else snapshot
        )
    if not train_task_stopped(state):
        failed = [process.returncode for process in processes if process.returncode != 0]
        if failed:
            raise RuntimeError(i18n("子进程执行失败，返回码：%s") % failed)


def run_preprocess_dataset(
    trainset_dir, exp_dir, sr, n_p, state, format_output=True, training_mode=None,
    noise_reduction=False, reduction_strength=0.75,
):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    if is_multispeaker_mode(training_mode):
        prepare_multispeaker_manifest(trainset_dir, exp_dir)
    log_path = "%s/logs/%s/preprocess.log" % (now_dir, exp_dir)
    with open(log_path, "w", encoding="utf8"):
        pass
    manifest_arg = (
        ' "%s/logs/%s/multispeaker_manifest.json"' % (now_dir, exp_dir)
        if is_multispeaker_mode(training_mode)
        else ""
    )
    noise_arg = (
        " %s %.2f" % ("true" if noise_reduction else "false", float(reduction_strength))
        if noise_reduction
        else ""
    )
    cmd = '"%s" train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f%s%s' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        config.preprocess_per,
        manifest_arg,
        noise_arg,
    )
    extract_start_time = time.time()
    requested_workers = max(int(n_p), 1)
    actual_workers = 1 if config.noparallel else requested_workers
    print(
        i18n(
            "数据提取开始：start_time=%.6f，请求并行数=%s，实际并行数上限=%s"
        )
        % (extract_start_time, requested_workers, actual_workers),
        flush=True,
    )
    try:
        process = start_train_process(state, cmd)
        yield from wait_train_processes(
            state, [process], log_path, "数据切分", format_output
        )
    finally:
        extract_end_time = time.time()
        print(
            i18n("数据提取结束：end_time=%.6f，总耗时=%.3f秒")
            % (extract_end_time, extract_end_time - extract_start_time),
            flush=True,
        )
    if not train_task_stopped(state):
        validate_preprocess_outputs(exp_dir)


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p, training_mode=None,
                        noise_reduction=False, reduction_strength=0.75):
    action, state = begin_train_task("数据切分")
    if action == "busy":
        yield (
            format_status(
                "数据切分",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("数据切分", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_preprocess_dataset(
            trainset_dir, exp_dir, sr, n_p, state, training_mode=training_mode,
            noise_reduction=noise_reduction, reduction_strength=reduction_strength,
        ):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("数据切分", "已停止")
    except Exception:
        final_info = format_status("数据切分", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("数据切分", "已完成")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_preprocess_dataset():
    return stop_train_task("数据切分")


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def run_extract_f0_feature(
    gpus,
    n_p,
    f0method,
    if_f0,
    exp_dir,
    gpus_rmvpe,
    state,
    format_output=True,
    embedder="contentvec",
):
    if f0method not in ("pm", "rmvpe", "nmvpe", "fcpe"):
        raise ValueError(i18n("仅支持pm、rmvpe、nmvpe和fcpe音高提取算法"))
    if embedder not in EMBEDDER_CHOICES:
        raise ValueError("Unsupported embedder: %s" % embedder)
    log_path = "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir)
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    validate_preprocess_outputs(exp_dir)
    with open(log_path, "w", encoding="utf8"):
        pass

    if if_f0:
        processes = []
        rmvpe_devices = [gpu for gpu in gpus_rmvpe.split("-") if gpu != ""]
        if f0method == "pm" or (
            f0method in ("rmvpe", "nmvpe", "fcpe")
            and not rmvpe_devices
            and not config.dml
        ):
            cmd = (
                '"%s" train/dataset/extract_f0.py cpu "%s/logs/%s" %s %s'
                % (config.python_cmd, now_dir, exp_dir, n_p, f0method)
            )
            processes.append(start_train_process(state, cmd))
        elif rmvpe_devices:
            count = len(rmvpe_devices)
            for index, gpu in enumerate(rmvpe_devices):
                cmd = (
                    '"%s" train/dataset/extract_f0.py cuda %s %s %s "%s/logs/%s" %s %s'
                    % (
                        config.python_cmd,
                        count,
                        index,
                        gpu,
                        now_dir,
                        exp_dir,
                        config.is_half,
                        f0method,
                    )
                )
                processes.append(start_train_process(state, cmd))
        else:
            cmd = (
                '"%s" train/dataset/extract_f0.py dml "%s/logs/%s" %s'
                % (config.python_cmd, now_dir, exp_dir, f0method)
            )
            processes.append(start_train_process(state, cmd))
        yield from wait_train_processes(
            state, processes, log_path, "F0提取", format_output
        )
        if train_task_stopped(state):
            return

        with open(log_path, "w", encoding="utf8"):
            pass

    feature_gpus = [gpu for gpu in gpus.split("-") if gpu != ""]
    processes = []
    if feature_gpus:
        count = len(feature_gpus)
        for index, gpu in enumerate(feature_gpus):
            cmd = (
                '"%s" train/dataset/extract_hubert_feature.py %s %s %s %s "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    config.device,
                    count,
                    index,
                    gpu,
                    now_dir,
                    exp_dir,
                    config.is_half,
                    embedder,
                )
            )
            processes.append(start_train_process(state, cmd))
    else:
        cmd = (
            '"%s" train/dataset/extract_hubert_feature.py %s 1 0 "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                now_dir,
                exp_dir,
                config.is_half,
                embedder,
            )
        )
        processes.append(start_train_process(state, cmd))
    yield from wait_train_processes(
        state, processes, log_path, "HuBERT特征", format_output
    )
    if not train_task_stopped(state):
        validate_feature_outputs(exp_dir, if_f0)


def extract_f0_feature(
    gpus, n_p, f0method, if_f0, exp_dir, gpus_rmvpe, embedder="contentvec"
):
    action, state = begin_train_task("特征提取")
    if action == "busy":
        yield (
            format_status(
                "特征提取",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("特征提取", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_extract_f0_feature(
            gpus,
            n_p,
            f0method,
            if_f0,
            exp_dir,
            gpus_rmvpe,
            state,
            embedder=embedder,
        ):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("特征提取", "已停止")
    except Exception:
        final_info = format_status("特征提取", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("特征提取", "已完成")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_extract_f0_feature():
    return stop_train_task("特征提取")

def get_pretrained_models(f0_str, sr2):
    root = "assets/pretrained_v2"
    if_pretrained_generator_exist = os.access(
        "%s/%sG%s.pth" % (root, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "%s/%sD%s.pth" % (root, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            i18n("生成器预训练模型不存在，将不使用：assets/pretrained_v2/%sG%s.pth"),
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            i18n("判别器预训练模型不存在，将不使用：assets/pretrained_v2/%sD%s.pth"),
            f0_str,
            sr2,
        )
    return (
        (
            "%s/%sG%s.pth" % (root, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "%s/%sD%s.pth" % (root, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3):
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(f0_str, sr2)


def change_f0(if_f0_3, sr2):  # f0method8,pretrained_G14,pretrained_D15
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": bool(if_f0_3) and F0GPUVisible, "__type__": "update"},
        *get_pretrained_models("f0" if if_f0_3 else "", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def run_train_model(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    state,
    format_output=True,
    training_mode=None,
    embedder="contentvec",
    ms_mel=True,
    spectral_detail=True,
    gradient_checkpointing=False,
    bf16=False,
):
    if embedder not in EMBEDDER_CHOICES:
        raise ValueError("Unsupported embedder: %s" % embedder)
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = "%s/3_feature768" % exp_dir
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    multi = is_multispeaker_mode(training_mode)
    manifest = load_experiment_manifest(exp_dir) if multi else None
    manifest_by_key = (
        {entry["output_key"]: entry for entry in manifest["entries"]}
        if manifest
        else {}
    )
    if multi:
        names = {
            name for name in names
            if name.rsplit("_", 1)[0] in manifest_by_key
        }
    if not names:
        raise RuntimeError(i18n("没有可用于训练的有效音频，请先完成数据切分和特征提取"))
    opt = []
    active_speaker_names = {}
    for name in sorted(names):
        entry = manifest_by_key.get(name.rsplit("_", 1)[0]) if multi else None
        speaker_id = entry["speaker_id"] if entry else int(spk_id5)
        if multi:
            speaker_name = entry["speaker_name"]
            active_speaker_names[speaker_id] = speaker_name
        repeat = entry["repeat"] if entry else 1
        if if_f0_3:
            line = (
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    speaker_id,
                )
            )
        else:
            line = (
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    speaker_id,
                )
            )
        if multi:
            line = "%s|%s" % (line, speaker_name)
        opt.extend([line] * repeat)
    fea_dim = 768
    mute_speaker_ids = (
        sorted(active_speaker_names)
        if multi
        else [int(spk_id5)]
    )
    if if_f0_3:
        for speaker_id in mute_speaker_ids:
            for _ in range(2):
                line = (
                    "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s|%s"
                    % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, speaker_id, active_speaker_names[speaker_id])
                    if multi
                    else "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                    % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, speaker_id)
                )
                opt.append(line)
    else:
        for speaker_id in mute_speaker_ids:
            for _ in range(2):
                line = (
                    "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s|%s"
                    % (now_dir, sr2, now_dir, fea_dim, speaker_id, active_speaker_names[speaker_id])
                    if multi
                    else "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                    % (now_dir, sr2, now_dir, fea_dim, speaker_id)
                )
                opt.append(line)
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w", encoding="utf8") as f:
        f.write("\n".join(opt))
    logger.debug(i18n("训练文件列表写入完成"))
    # 生成config#无需生成config
    logger.info(i18n("使用显卡：%s"), str(gpus16))
    if pretrained_G14 == "":
        logger.info(i18n("未使用生成器预训练模型"))
    if pretrained_D15 == "":
        logger.info(i18n("未使用判别器预训练模型"))
    config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if pathlib.Path(config_save_path).exists():
        config_data = json.loads(read_text(config_save_path))
    else:
        config_data = copy.deepcopy(config.json_config[config_path])
    if multi:
        config_data["model"]["spk_embed_dim"] = 110
        config_data["speaker_info"] = [
            {"id": speaker_id, "name": active_speaker_names[speaker_id]}
            for speaker_id in sorted(active_speaker_names)
        ]
    else:
        config_data.pop("speaker_info", None)
    config_data["embedder_model"] = embedder
    with open(config_save_path, "w", encoding="utf8") as f:
        json.dump(
            config_data,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )
        f.write("\n")
    extra_flags = ""
    if ms_mel:
        extra_flags += " --ms-mel 1"
    if spectral_detail:
        extra_flags += " --spectral-detail 1"
    if gradient_checkpointing:
        extra_flags += " --grad-ckpt 1"
    if bf16:
        extra_flags += " --bf16 1"
    extra_flags += ' --embedder "%s"' % embedder
    if gpus16:
        cmd = (
            '"%s" train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s%s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                extra_flags,
            )
        )
    else:
        cmd = (
            '"%s" train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s%s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                extra_flags,
            )
        )
    logger.info("%s: %s", i18n("执行命令"), cmd)
    process = start_train_process(state, cmd)
    yield from wait_train_processes(
        state,
        [process],
        os.path.join(exp_dir, "train.log"),
        "模型训练",
        format_output,
        True,
    )


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    training_mode=None,
    embedder="contentvec",
    ms_mel=True,
    spectral_detail=True,
    gradient_checkpointing=False,
    bf16=False,
):
    known_models = tuple(weight_names())
    action, state = begin_train_task("模型训练")
    if action == "busy":
        yield (
            format_status(
                "模型训练",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("模型训练", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
            button_update(),
        )
        for info in run_train_model(
            exp_dir1,
            sr2,
            if_f0_3,
            spk_id5,
            save_epoch10,
            total_epoch11,
            batch_size12,
            if_save_latest13,
            pretrained_G14,
            pretrained_D15,
            gpus16,
            if_cache_gpu17,
            if_save_every_weights18,
            state,
            training_mode=training_mode,
            embedder=embedder,
            ms_mel=ms_mel,
            spectral_detail=spectral_detail,
            gradient_checkpointing=gradient_checkpointing,
            bf16=bf16,
        ):
            known_models, model_update = refresh_weight_choices(known_models)
            yield (
                info,
                button_update(visible=False),
                button_update(visible=True),
                model_update,
            )
        if train_task_stopped(state):
            final_info = format_status("模型训练", "已停止")
    except Exception:
        final_info = format_status("模型训练", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("模型训练", "已完成")
    model_update = change_choices()
    yield (
        final_info,
        button_update(visible=True),
        button_update(visible=False),
        model_update,
    )


def stop_train_model():
    return stop_train_task("模型训练")


# but4.click(train_index, [exp_dir1], info3)
def run_train_index(exp_dir1, state, format_output=True, training_mode=None):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "train_index.log")
    with open(log_path, "w", encoding="utf8"):
        pass
    index_mode = (
        "multi"
        if is_multispeaker_mode(training_mode)
        else "single"
        if training_mode
        else "auto"
    )
    cmd = (
        '"%s" train/train_index.py "%s" "%s" %s %s'
        % (
            config.python_cmd,
            exp_dir1,
            outside_index_root,
            config.n_cpu,
            index_mode,
        )
    )
    process = start_train_process(state, cmd)
    yield from wait_train_processes(
        state, [process], log_path, "索引训练", format_output
    )


def train_index(exp_dir1, training_mode=None):
    action, state = begin_train_task("索引训练")
    if action == "busy":
        yield (
            format_status(
                "索引训练",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("索引训练", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_train_index(exp_dir1, state, True, training_mode):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("索引训练", "已停止")
    except Exception:
        final_info = format_status("索引训练", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("索引训练", "已完成")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_train_index():
    return stop_train_task("索引训练")

# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    gpus_rmvpe,
    training_mode=None,
    embedder="contentvec",
    noise_reduction=False,
    reduction_strength=0.75,
    ms_mel=True,
    spectral_detail=True,
    gradient_checkpointing=False,
    bf16=False,
):
    known_models = tuple(weight_names())
    action, state = begin_train_task("一键训练")
    if action == "busy":
        yield (
            format_status(
                "一键训练",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
            button_update(),
        )
        return

    completed_steps = []
    step = ""
    final_info = None
    running = True
    start_button = button_update(visible=False)
    stop_button = button_update(visible=True)
    try:
        yield (
            format_status("一键训练", "正在启动"),
            start_button,
            stop_button,
            button_update(),
        )

        step = "数据切分"
        yield (
            format_workflow_status(step, completed_steps=completed_steps),
            start_button,
            stop_button,
            button_update(),
        )
        for info in run_preprocess_dataset(
            trainset_dir4,
            exp_dir1,
            sr2,
            np7,
            state,
            False,
            training_mode,
            noise_reduction=noise_reduction,
            reduction_strength=reduction_strength,
        ):
            yield (
                format_workflow_status(step, info, completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
        running = not train_task_stopped(state)
        if running:
            completed_steps.append(step)

        if running:
            step = "F0与HuBERT特征提取"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_extract_f0_feature(
                gpus16,
                np7,
                f0method8,
                if_f0_3,
                exp_dir1,
                gpus_rmvpe,
                state,
                False,
                embedder=embedder,
            ):
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    button_update(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if running:
            step = "模型训练"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_train_model(
                exp_dir1,
                sr2,
                if_f0_3,
                spk_id5,
                save_epoch10,
                total_epoch11,
                batch_size12,
                if_save_latest13,
                pretrained_G14,
                pretrained_D15,
                gpus16,
                if_cache_gpu17,
                if_save_every_weights18,
                state,
                False,
                training_mode,
                embedder=embedder,
                ms_mel=ms_mel,
                spectral_detail=spectral_detail,
                gradient_checkpointing=gradient_checkpointing,
                bf16=bf16,
            ):
                known_models, model_update = refresh_weight_choices(known_models)
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    model_update,
                )
            if not train_task_stopped(state):
                yield (
                    format_workflow_status(step, completed_steps=completed_steps),
                    start_button,
                    stop_button,
                    change_choices(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if running:
            step = "索引训练"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_train_index(exp_dir1, state, False, training_mode):
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    button_update(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if not running:
            final_info = format_workflow_status(
                step, completed_steps=completed_steps, state="已停止"
            )
        else:
            final_info = format_workflow_status(
                "", completed_steps=completed_steps, state="已完成"
            )
    except Exception:
        final_info = format_workflow_status(
            step,
            traceback.format_exc(),
            completed_steps,
            "失败",
        )
    finally:
        finish_train_task(state)
    model_update = change_choices()
    yield (
        final_info,
        button_update(visible=True),
        button_update(visible=False),
        model_update,
    )


def stop_train1key():
    return stop_train_task("一键训练")

#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}
    try:
        info = eval(
            read_text(
                ckpt_path.replace(os.path.basename(ckpt_path), "train.log")
            )
            .strip("\n")
            .split("\n")[0]
            .split("\t")[-1]
        )
        sr, f0 = info["sample_rate"], info["if_f0"]
        return sr, str(f0)
    except Exception:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = IS_GPU


def main():
    import sys

    from tools.studio_app import launch_studio

    launch_studio(config, sys.modules[__name__])


if __name__ == "__main__":
    main()
