import json
import os
import re
import shutil
import socket
import threading
import time
import traceback
import uuid
import webbrowser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from infer.vc.utils import get_index_paths_from_model


core = None
STATIC_DIR = Path(__file__).resolve().parent / "studio_static"
UPLOAD_DIR = Path(os.environ.get("TEMP", "TEMP")) / "studio-uploads"
OUTPUT_DIR = Path(os.environ.get("TEMP", "TEMP")) / "studio-outputs"
DATASET_DIR = Path(os.environ.get("TEMP", "TEMP")) / "studio-datasets"
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


def _safe_extract_zip(archive, destination):
    import zipfile

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    with zipfile.ZipFile(str(archive), "r") as zf:
        for member in zf.infolist():
            name = member.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            target = (destination / name).resolve()
            if os.path.commonpath([str(destination), str(target)]) != str(destination):
                raise ValueError("The zip archive contains an unsafe path")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("The zip archive contains a symbolic link")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted_files.append(target)
    return extracted_files


def _dataset_root(extracted_root):
    root = Path(extracted_root)
    children = list(root.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return root


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


def _request(url, method="GET", headers=None, data=None, timeout=60):
    req = Request(url, data=data, method=method, headers=headers or {})
    req.add_header("User-Agent", "NVC-Studio/1.0")
    return urlopen(req, timeout=timeout)


def _copy_stream(response, dest, on_progress=None):
    total = int(response.headers.get("Content-Length") or 0)
    done = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if on_progress:
                on_progress(done, total)
    return done, total


def _filename_from_url(url, fallback="model.bin"):
    name = Path(unquote(urlparse(url).path)).name
    return name or fallback


def _gdrive_id(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if query.get("id"):
        return query["id"][0]
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", parsed.path or "")
    return match.group(1) if match else None


def _download_gdrive(url, dest, on_progress=None):
    file_id = _gdrive_id(url)
    if not file_id:
        raise ValueError("Google Drive link is missing a file id")
    session_url = "https://drive.google.com/uc?export=download&id=%s" % file_id
    response = _request(session_url)
    html = ""
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        html = response.read().decode("utf-8", "ignore")
        response.close()
        confirm = re.search(r"confirm=([0-9A-Za-z_]+)", html)
        if not confirm:
            raise RuntimeError("Google Drive did not return a direct file. Check sharing.")
        session_url = (
            "https://drive.google.com/uc?export=download&confirm=%s&id=%s"
            % (confirm.group(1), file_id)
        )
        response = _request(session_url)
    name = dest.name
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        dest = dest.with_name(match.group(1))
    _copy_stream(response, dest, on_progress)
    response.close()
    return dest


def _download_yandex(url, dest, on_progress=None):
    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=%s" % url
    with _request(api) as response:
        payload = json.loads(response.read().decode("utf-8"))
    href = payload.get("href")
    if not href:
        raise RuntimeError("Yandex Disk did not return a download href")
    with _request(href) as response:
        name = dest.name
        disposition = response.headers.get("Content-Disposition", "")
        match = re.search(r"filename\*=UTF-8''([^;]+)|filename=\"?([^\";]+)\"?", disposition)
        if match:
            dest = dest.with_name(unquote(match.group(1) or match.group(2)))
        _copy_stream(response, dest, on_progress)
    return dest


class _HrefParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def _download_mediafire(url, dest, on_progress=None):
    with _request(url) as page:
        html = page.read().decode("utf-8", "ignore")
    parser = _HrefParser()
    parser.feed(html)
    direct = next(
        (href for href in parser.hrefs if "download" in href and "mediafire.com" in href),
        None,
    )
    if not direct:
        raise RuntimeError("MediaFire direct link was not found")
    with _request(direct) as response:
        disposition = response.headers.get("Content-Disposition", "")
        match = re.search(r'filename="?([^";]+)"?', disposition)
        if match:
            dest = dest.with_name(match.group(1))
        _copy_stream(response, dest, on_progress)
    return dest


def _download_hf(url, dest, on_progress=None):
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if parsed.netloc.endswith("huggingface.co") and "/resolve/" in path:
        with _request(url) as response:
            dest = dest.with_name(_filename_from_url(url, dest.name))
            _copy_stream(response, dest, on_progress)
        return dest
    from huggingface_hub import hf_hub_download

    if parsed.netloc.endswith("huggingface.co"):
        parts = [part for part in path.split("/") if part]
        if "blob" in parts:
            idx = parts.index("blob")
            repo = "/".join(parts[:2])
            revision = parts[idx + 1]
            filename = "/".join(parts[idx + 2:])
        elif len(parts) >= 2:
            repo = "/".join(parts[:2])
            revision = "main"
            filename = dest.name
        else:
            raise ValueError("Unsupported Hugging Face URL")
        saved = hf_hub_download(repo, filename, revision=revision, local_dir=str(dest.parent))
        return Path(saved)
    repo = url.strip()
    saved = hf_hub_download(repo, dest.name, revision="main", local_dir=str(dest.parent))
    return Path(saved)


def _download_mega(url, dest, on_progress=None):
    try:
        from mega import Mega
    except ImportError as error:
        raise RuntimeError("Mega downloads need the mega.py package") from error
    client = Mega()
    path = client.download_url(url, dest_path=str(dest.parent), dest_filename=dest.name)
    if on_progress:
        size = Path(path).stat().st_size
        on_progress(size, size)
    return Path(path)


def detect_source(url):
    host = urlparse(url).netloc.lower()
    if "drive.google.com" in host or "docs.google.com" in host:
        return "gdrive"
    if "mega.nz" in host or "mega.co.nz" in host:
        return "mega"
    if "disk.yandex" in host or "yadi.sk" in host:
        return "yandex"
    if "mediafire.com" in host:
        return "mediafire"
    if "huggingface.co" in host or "hf.co" in host:
        return "hf"
    return "direct"


def resolve_library_dest(kind, filename):
    name = Path(filename or "download.bin").name
    roots = {
        "weights": Path(os.environ.get("weight_root", "assets/weights")),
        "indices": Path(os.environ.get("outside_index_root", "assets/indices")),
        "pretrained": Path("assets/pretrained_v2"),
        "zip": Path(os.environ.get("weight_root", "assets/weights")),
        "custom": Path(os.environ.get("TEMP", "TEMP")) / "studio-library",
    }
    root = roots.get(kind, roots["weights"])
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def download_library_file(url, dest, source=None, on_progress=None):
    source = source or detect_source(url)
    handlers = {
        "gdrive": _download_gdrive,
        "yandex": _download_yandex,
        "mediafire": _download_mediafire,
        "hf": _download_hf,
        "mega": _download_mega,
    }
    handler = handlers.get(source)
    if handler:
        return handler(url, dest, on_progress)
    dest = dest.with_name(_filename_from_url(url, dest.name))
    with _request(url) as response:
        _copy_stream(response, dest, on_progress)
    return dest


def _pretrained_choices(sr, if_f0, version):
    root = Path("assets/pretrained_v2" if version == "v2" else "assets/pretrained")
    root = root.resolve()
    if not root.is_dir():
        return [], []
    prefix = "f0" if _as_bool(if_f0, True) else ""
    generator_suffix = (prefix + "g" + str(sr)).lower()
    discriminator_suffix = (prefix + "d" + str(sr)).lower()
    generator = []
    discriminator = []
    for path in sorted(root.rglob("*.pth")):
        stem = path.stem.lower()
        relative = path.relative_to(Path.cwd()).as_posix() if path.is_relative_to(Path.cwd()) else path.as_posix()
        if stem == generator_suffix:
            generator.append(relative)
        elif stem == discriminator_suffix:
            discriminator.append(relative)
    return generator, discriminator


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

    @app.post("/api/library/import")
    def library_import(payload: dict):
        import zipfile

        url = str(payload.get("url") or "").strip()
        if not url:
            return _error("A download URL is required")
        kind = payload.get("kind") or "zip"
        source = payload.get("source") or "auto"
        if source in {"", "auto"}:
            source = detect_source(url)
        filename = payload.get("filename") or _filename_from_url(url, "model.zip" if kind == "zip" else "model.bin")
        dest = resolve_library_dest(kind, filename)
        try:
            saved = download_library_file(url, dest, source=source)
        except Exception as error:
            return _error(str(error), 400)
        extracted = []
        if kind == "zip" and str(saved).lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(str(saved), "r") as zf:
                    zf.extractall(str(saved.parent))
                    extracted = [str(saved.parent / n) for n in zf.namelist() if not n.endswith("/")]
            except zipfile.BadZipFile:
                return _error("Downloaded file is not a valid zip archive", 400)
        return _ok(
            path=str(saved),
            name=saved.name,
            source=source,
            kind=kind,
            extracted=extracted,
            models=core.weight_names(),
        )

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
            index_choices=get_index_paths_from_model(model),
        )

    @app.post("/api/infer/speaker-index")
    def infer_speaker_index(payload: dict):
        index1, index2 = core.update_speaker_index(
            payload.get("model") or "",
            payload.get("speaker"),
            payload.get("speaker_label"),
        )
        return _ok(
            index=index1.get("value", ""),
            index_batch=index2.get("value", ""),
            index_choices=get_index_paths_from_model(payload.get("model") or ""),
        )

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

    @app.post("/api/train/dataset")
    async def train_dataset(dataset: UploadFile = File(...)):
        filename = Path(dataset.filename or "dataset.zip").name
        if not filename.lower().endswith(".zip"):
            return _error("Only .zip dataset archives are supported")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        dataset_id = uuid.uuid4().hex
        archive = DATASET_DIR / (dataset_id + ".zip")
        destination = DATASET_DIR / dataset_id
        try:
            with archive.open("wb") as handle:
                shutil.copyfileobj(dataset.file, handle)
            extracted = _safe_extract_zip(archive, destination)
            audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
            if not any(path.suffix.lower() in audio_extensions for path in extracted):
                shutil.rmtree(destination, ignore_errors=True)
                return _error("The archive does not contain audio files")
            return _ok(
                path=str(_dataset_root(destination)),
                name=filename,
                files=len(extracted),
            )
        except Exception as error:
            shutil.rmtree(destination, ignore_errors=True)
            return _error(str(error))
        finally:
            archive.unlink(missing_ok=True)

    @app.post("/api/train/mode")
    def train_mode(payload: dict):
        folder, speaker = core.change_training_mode(payload.get("mode"))
        return _ok(folder=folder, speaker=speaker)

    @app.post("/api/train/sr")
    def train_sr(payload: dict):
        g, d = core.change_sr2(payload.get("sr"), payload.get("if_f0"), payload.get("version"))
        return _ok(pretrained_g=g, pretrained_d=d)

    @app.get("/api/train/pretrained")
    def train_pretrained(sr: str = "40k", if_f0: str = "1", version: str = "v2"):
        generator, discriminator = _pretrained_choices(sr, if_f0, version)
        preferred_g, preferred_d = core.change_sr2(sr, _as_bool(if_f0, True), version)
        if preferred_g and preferred_g not in generator:
            generator.insert(0, preferred_g)
        if preferred_d and preferred_d not in discriminator:
            discriminator.insert(0, preferred_d)
        return _ok(
            generator=[""] + generator,
            discriminator=[""] + discriminator,
            pretrained_g=preferred_g,
            pretrained_d=preferred_d,
        )

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
