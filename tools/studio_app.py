import json
import os
import shutil
import socket
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


core = None
STATIC_DIR = Path(__file__).resolve().parent / "studio_static"
UPLOAD_DIR = Path(os.environ.get("TEMP", "TEMP")) / "studio-uploads"
OUTPUT_DIR = Path(os.environ.get("TEMP", "TEMP")) / "studio-outputs"
FAQ_FILES = {
    "zh": "docs/cn/faq.md",
    "en": "docs/en/faq_en.md",
    "ru": "docs/en/faq_en.md",
}


def _json(data, status=200):
    return JSONResponse(data, status_code=status)


def _error(message, status=400):
    return _json({"ok": False, "error": str(message)}, status)


def _ok(**payload):
    return _json({"ok": True, **payload})


def _save_upload(upload: UploadFile, prefix="input"):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    path = UPLOAD_DIR / ("%s-%s%s" % (prefix, uuid.uuid4().hex, suffix))
    with path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return str(path)


def _save_uploads(uploads):
    return [_save_upload(item, "batch") for item in uploads or [] if item and item.filename]


def _write_audio(audio, sample_rate):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / ("out-%s.wav" % uuid.uuid4().hex)
    sf.write(str(path), np.asarray(audio), int(sample_rate))
    return str(path)


def _as_bool(value, default=False):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "是"}


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _last_text(result):
    if isinstance(result, tuple):
        return result[0] if result else ""
    return result


def _iter_text(generator):
    last = ""
    for item in generator:
        if isinstance(item, tuple):
            last = item[0]
        else:
            last = item
        yield last
    return last


def _sse(events):
    def stream():
        try:
            for event in events:
                payload = event if isinstance(event, dict) else {"text": str(event)}
                yield "data: %s\n\n" % json.dumps(payload, ensure_ascii=False)
        except Exception as error:
            yield "data: %s\n\n" % json.dumps(
                {"done": True, "ok": False, "text": traceback.format_exc() or str(error)},
                ensure_ascii=False,
            )
            return
        yield "data: %s\n\n" % json.dumps({"done": True, "ok": True}, ensure_ascii=False)

    return StreamingResponse(stream(), media_type="text/event-stream")


def bind_core(module):
    global core
    core = module
    return module


def create_app(core_module=None):
    if core_module is not None:
        bind_core(core_module)
    if core is None:
        raise RuntimeError("Studio core is not bound")
    app = FastAPI(title="NVC Studio")
    app.mount("/assets/static", StaticFiles(directory=str(STATIC_DIR)), name="studio-static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        pretrained_g, pretrained_d = core.change_sr2("40k", True, "v2")
        return _ok(
            device=str(core.config.device),
            dtype=str(core.config.dtype),
            gpu_ok=bool(core.IS_GPU),
            gpu_info=core.gpu_info,
            gpus=core.gpus,
            feature_gpus=core.feature_gpus,
            n_cpu=int(core.config.n_cpu),
            default_batch_size=int(core.default_batch_size),
            default_f0=core.default_training_f0_method,
            f0_gpu_visible=bool(core.F0GPUVisible),
            models=core.weight_names(),
            pymss_models=list(core.PYMSS_MODEL_CHOICES),
            pymss_info=core.get_model_info(core.PYMSS_MODEL_CHOICES[0]),
            pretrained_g=pretrained_g,
            pretrained_d=pretrained_d,
        )

    @app.get("/api/models")
    def models():
        return _ok(models=core.weight_names())

    @app.post("/api/models/unload")
    def unload_model():
        core.vc.get_vc("", 0.33, 0.33)
        return _ok(models=core.weight_names())

    @app.get("/api/faq")
    def faq(lang: str = "en"):
        path = FAQ_FILES.get(lang, FAQ_FILES["en"])
        try:
            return _ok(markdown=core.read_text(path))
        except Exception:
            return _ok(markdown=traceback.format_exc())

    @app.get("/api/pymss/info")
    def pymss_info(model: str):
        return _ok(info=core.get_model_info(model))

    @app.post("/api/infer/select")
    def infer_select(payload: dict):
        model = payload.get("model") or ""
        protect0 = _as_float(payload.get("protect0"), 0.33)
        protect1 = _as_float(payload.get("protect1"), 0.33)
        slider, dropdown, p0, p1, index1, index2 = core.vc.get_vc(model, protect0, protect1)
        return _ok(
            speaker_slider=slider,
            speaker_dropdown=dropdown,
            protect0=p0,
            protect1=p1,
            index=index1.get("value", ""),
            index_batch=index2.get("value", ""),
        )

    @app.post("/api/infer/speaker-index")
    def infer_speaker_index(payload: dict):
        index1, index2 = core.update_speaker_index(
            payload.get("model") or "",
            payload.get("speaker"),
            payload.get("speaker_label"),
        )
        return _ok(index=index1.get("value", ""), index_batch=index2.get("value", ""))

    @app.post("/api/infer/single")
    async def infer_single(
        speaker: str = Form("0"),
        speaker_label: str = Form(""),
        pitch: str = Form("0"),
        f0_method: str = Form("rmvpe"),
        index_path: str = Form(""),
        index_rate: str = Form("0.75"),
        resample_sr: str = Form("0"),
        rms_mix_rate: str = Form("0.25"),
        protect: str = Form("0.33"),
        audio: UploadFile = File(...),
    ):
        try:
            core.report_missing_index(index_path)
        except Exception as error:
            return _error(getattr(error, "args", [error])[0])
        path = _save_upload(audio)
        status, audio_out = core.vc_single_with_speaker(
            speaker,
            speaker_label or None,
            path,
            _as_int(pitch),
            f0_method,
            index_path,
            _as_float(index_rate, 0.75),
            _as_int(resample_sr),
            _as_float(rms_mix_rate, 0.25),
            _as_float(protect, 0.33),
        )
        if not audio_out or audio_out[0] is None or audio_out[1] is None:
            return _ok(status=status, audio=None)
        out_path = _write_audio(audio_out[1], audio_out[0])
        return _ok(status=status, audio="/api/file?path=%s" % out_path)

    @app.post("/api/infer/batch")
    async def infer_batch(request: Request):
        form = await request.form()
        files = [item for item in form.getlist("files") if hasattr(item, "filename")]
        paths = _save_uploads(files)
        try:
            core.report_missing_index(form.get("index_path"))
        except Exception as error:
            return _error(getattr(error, "args", [error])[0])

        def events():
            for text in core.vc_multi_with_speaker(
                form.get("speaker") or 0,
                form.get("speaker_label") or None,
                form.get("input_dir") or "",
                form.get("output_dir") or "opt",
                paths,
                _as_int(form.get("pitch")),
                form.get("f0_method") or "rmvpe",
                form.get("index_path") or "",
                _as_float(form.get("index_rate"), 1.0),
                _as_int(form.get("resample_sr")),
                _as_float(form.get("rms_mix_rate"), 1.0),
                _as_float(form.get("protect"), 0.33),
                form.get("format") or "wav",
            ):
                yield {"text": text}

        return _sse(events())

    @app.post("/api/separate")
    async def separate(request: Request):
        form = await request.form()
        files = [item for item in form.getlist("files") if hasattr(item, "filename")]
        paths = _save_uploads(files)

        def events():
            for info, progress, start, stop in core.run_pymss_separation(
                form.get("model") or core.PYMSS_MODEL_CHOICES[0],
                form.get("input_dir") or "",
                form.get("vocal_dir") or "opt",
                paths,
                form.get("residual_dir") or "opt",
                form.get("format") or "flac",
            ):
                yield {
                    "text": info,
                    "progress": progress,
                    "start_visible": start.get("visible", True),
                    "stop_visible": stop.get("visible", False),
                }

        return _sse(events())

    @app.post("/api/separate/stop")
    def separate_stop():
        info, progress, start, stop = core.stop_pymss_gui()
        return _ok(
            text=info,
            progress=progress,
            start_visible=start.get("visible", True),
            stop_visible=stop.get("visible", False),
        )

    @app.post("/api/train/mode")
    def train_mode(payload: dict):
        folder, speaker = core.change_training_mode(payload.get("mode"))
        return _ok(folder=folder, speaker=speaker)

    @app.post("/api/train/sr")
    def train_sr(payload: dict):
        g, d = core.change_sr2(payload.get("sr"), payload.get("if_f0"), payload.get("version"))
        return _ok(pretrained_g=g, pretrained_d=d)

    @app.post("/api/train/version")
    def train_version(payload: dict):
        g, d, sr = core.change_version19(
            payload.get("sr"), payload.get("if_f0"), payload.get("version")
        )
        return _ok(pretrained_g=g, pretrained_d=d, sr=sr)

    @app.post("/api/train/f0")
    def train_f0(payload: dict):
        method, rmvpe, g, d = core.change_f0(
            payload.get("if_f0"), payload.get("sr"), payload.get("version")
        )
        return _ok(
            f0_visible=method.get("visible", True),
            rmvpe_visible=rmvpe.get("visible", True),
            pretrained_g=g,
            pretrained_d=d,
        )

    def _yesno(value, default=False):
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on", "是", "да"}:
            return core.i18n("是")
        if text in {"0", "false", "no", "off", "否", "нет"}:
            return core.i18n("否")
        return core.i18n("是") if default else core.i18n("否")

    def _train_form(form):
        mode = form.get("mode") or ""
        multi = mode in {"多说话人", "Multiple speakers", "Несколько спикеров"}
        return {
            "exp": form.get("exp") or "",
            "sr": form.get("sr") or "40k",
            "if_f0": _as_bool(form.get("if_f0"), True),
            "version": form.get("version") or "v2",
            "n_p": _as_int(form.get("n_p"), 4),
            "mode": core.i18n("多说话人") if multi else core.i18n("单说话人"),
            "trainset": form.get("trainset") or "",
            "spk_id": _as_int(form.get("spk_id")),
            "gpus": form.get("gpus") or core.feature_gpus,
            "gpus_rmvpe": form.get("gpus_rmvpe") or core.feature_gpus,
            "f0_method": form.get("f0_method") or core.default_training_f0_method,
            "save_epoch": _as_int(form.get("save_epoch"), 5),
            "total_epoch": _as_int(form.get("total_epoch"), 20),
            "batch_size": _as_int(form.get("batch_size"), core.default_batch_size),
            "save_latest": _yesno(form.get("save_latest")),
            "cache_gpu": _yesno(form.get("cache_gpu")),
            "save_every": _yesno(form.get("save_every")),
            "pretrained_g": form.get("pretrained_g") or "",
            "pretrained_d": form.get("pretrained_d") or "",
            "train_gpus": form.get("train_gpus") or core.gpus,
        }

    @app.post("/api/train/preprocess")
    async def train_preprocess(request: Request):
        data = _train_form(await request.form())

        def events():
            for info, start, stop in core.preprocess_dataset(
                data["trainset"], data["exp"], data["sr"], data["n_p"], data["mode"]
            ):
                yield {
                    "text": info,
                    "start_visible": start.get("visible", True),
                    "stop_visible": stop.get("visible", False),
                }

        return _sse(events())

    @app.post("/api/train/preprocess/stop")
    def train_preprocess_stop():
        info, start, stop = core.stop_preprocess_dataset()
        return _ok(text=info, start_visible=start.get("visible", True), stop_visible=stop.get("visible", False))

    @app.post("/api/train/extract")
    async def train_extract(request: Request):
        data = _train_form(await request.form())

        def events():
            for info, start, stop in core.extract_f0_feature(
                data["gpus"],
                data["n_p"],
                data["f0_method"],
                data["if_f0"],
                data["exp"],
                data["version"],
                data["gpus_rmvpe"],
            ):
                yield {
                    "text": info,
                    "start_visible": start.get("visible", True),
                    "stop_visible": stop.get("visible", False),
                }

        return _sse(events())

    @app.post("/api/train/extract/stop")
    def train_extract_stop():
        info, start, stop = core.stop_extract_f0_feature()
        return _ok(text=info, start_visible=start.get("visible", True), stop_visible=stop.get("visible", False))

    @app.post("/api/train/model")
    async def train_model(request: Request):
        data = _train_form(await request.form())

        def events():
            for info, start, stop, models in core.click_train(
                data["exp"],
                data["sr"],
                data["if_f0"],
                data["spk_id"],
                data["save_epoch"],
                data["total_epoch"],
                data["batch_size"],
                data["save_latest"],
                data["pretrained_g"],
                data["pretrained_d"],
                data["train_gpus"],
                data["cache_gpu"],
                data["save_every"],
                data["version"],
                data["mode"],
            ):
                yield {
                    "text": info,
                    "start_visible": start.get("visible", True),
                    "stop_visible": stop.get("visible", False),
                    "models": models.get("choices") if isinstance(models, dict) else None,
                }

        return _sse(events())

    @app.post("/api/train/model/stop")
    def train_model_stop():
        info, start, stop = core.stop_train_model()
        return _ok(text=info, start_visible=start.get("visible", True), stop_visible=stop.get("visible", False))

    @app.post("/api/train/index")
    async def train_index(request: Request):
        data = _train_form(await request.form())

        def events():
            for info, start, stop in core.train_index(data["exp"], data["version"], data["mode"]):
                yield {
                    "text": info,
                    "start_visible": start.get("visible", True),
                    "stop_visible": stop.get("visible", False),
                }

        return _sse(events())

    @app.post("/api/train/index/stop")
    def train_index_stop():
        info, start, stop = core.stop_train_index()
        return _ok(text=info, start_visible=start.get("visible", True), stop_visible=stop.get("visible", False))

    @app.post("/api/train/oneclick")
    async def train_oneclick(request: Request):
        data = _train_form(await request.form())

        def events():
            for info, start, stop, models in core.train1key(
                data["exp"],
                data["sr"],
                data["if_f0"],
                data["trainset"],
                data["spk_id"],
                data["n_p"],
                data["f0_method"],
                data["save_epoch"],
                data["total_epoch"],
                data["batch_size"],
                data["save_latest"],
                data["pretrained_g"],
                data["pretrained_d"],
                data["train_gpus"],
                data["cache_gpu"],
                data["save_every"],
                data["version"],
                data["gpus_rmvpe"],
                data["mode"],
            ):
                yield {
                    "text": info,
                    "start_visible": start.get("visible", True),
                    "stop_visible": stop.get("visible", False),
                    "models": models.get("choices") if isinstance(models, dict) else None,
                }

        return _sse(events())

    @app.post("/api/train/oneclick/stop")
    def train_oneclick_stop():
        info, start, stop = core.stop_train1key()
        return _ok(text=info, start_visible=start.get("visible", True), stop_visible=stop.get("visible", False))

    @app.post("/api/helper/submit")
    def helper_submit(payload: dict):
        rows = payload.get("rows") or []
        padded = core.empty_multispeaker_rows()
        for index, row in enumerate(rows[: core.MULTISPEAKER_MAX_ROWS]):
            padded[index] = [
                row.get("path", ""),
                row.get("name", ""),
                row.get("id", ""),
                row.get("repeat", ""),
            ]
        try:
            stored, status = core.submit_multispeaker_rows(
                payload.get("exp") or "",
                padded,
                max(2, len(rows)),
                0,
            )
        except Exception as error:
            return _error(getattr(error, "args", [error])[0])
        return _ok(status=status, rows=stored[: max(2, len(rows))])

    @app.post("/api/ckpt/merge")
    def ckpt_merge(payload: dict):
        return _ok(
            text=core.merge(
                payload.get("a") or "",
                payload.get("b") or "",
                _as_float(payload.get("alpha"), 0.5),
                payload.get("sr") or "40k",
                payload.get("if_f0") or core.i18n("是"),
                payload.get("info") or "",
                payload.get("name") or "",
                payload.get("version") or "v1",
            )
        )

    @app.post("/api/ckpt/modify")
    def ckpt_modify(payload: dict):
        return _ok(
            text=core.change_info(
                payload.get("path") or "",
                payload.get("info") or "",
                payload.get("name") or "",
            )
        )

    @app.post("/api/ckpt/show")
    def ckpt_show(payload: dict):
        return _ok(text=core.show_info(payload.get("path") or ""))

    @app.post("/api/ckpt/extract-info")
    def ckpt_extract_info(payload: dict):
        sr, f0, version = core.change_info_(payload.get("path") or "")
        return _ok(sr=sr, f0=f0, version=version)

    @app.post("/api/ckpt/extract")
    def ckpt_extract(payload: dict):
        return _ok(
            text=core.extract_small_model(
                payload.get("path") or "",
                payload.get("name") or "",
                payload.get("sr") or "40k",
                payload.get("if_f0") or "1",
                payload.get("info") or "",
                payload.get("version") or "v2",
            )
        )

    @app.get("/api/file")
    def file_get(path: str):
        target = Path(path).resolve()
        allowed = (
            OUTPUT_DIR.resolve(),
            Path(os.environ.get("TEMP", "TEMP")).resolve(),
            Path.cwd().resolve() / "opt",
        )
        if not any(str(target).startswith(str(root)) for root in allowed):
            return _error("File is outside the allowed output directories", 403)
        if not target.is_file():
            return _error("File not found", 404)
        return FileResponse(target)

    return app


def launch_studio(config, core_module=None):
    import uvicorn

    bind_core(core_module)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    port = core.find_available_port(config.listen_port)
    config.listen_port = port
    url = "http://127.0.0.1:%s" % port
    print("NVC Studio: %s" % url, flush=True)
    print("NVC_STUDIO_PORT=%s" % port, flush=True)
    if config.iscolab:
        print(
            "Colab: the notebook kernel must proxy this port after the server is listening.",
            flush=True,
        )
    elif not config.noautoopen:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        create_app(core),
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
