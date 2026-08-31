import logging
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoFeatureExtractor, HubertModel

from tools.cuda_graph import run_cuda_graph


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


HUBERT_MODEL_PATH = (PROJECT_ROOT / "assets" / "hubert_base").resolve()

# Embedder registry. hubert_base ships with NVC; the remaining models are
# downloaded from their upstream Hugging Face repositories on first use.
EMBEDDER_CHOICES = (
    "hubert_base",
    "contentvec",
    "spin",
    "spin-v2",
    "mhubert147",
    "mhubert147-phonetic",
)
EMBEDDER_ALIASES = {"mhubert147-phonetic": "mhubert147"}
PHONETIC_LAYER_MIX = ((6, 0.15), (9, 0.70), (12, 0.15))
APPLIO_EMBEDDER_DOWNLOAD_URL = (
    "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders"
)
EMBEDDER_DOWNLOAD_SPECS = {
    name: {
        "base_url": "%s/%s" % (APPLIO_EMBEDDER_DOWNLOAD_URL, name),
        "files": ("pytorch_model.bin", "config.json"),
    }
    for name in ("contentvec", "spin", "spin-v2")
}
EMBEDDER_DOWNLOAD_SPECS["mhubert147"] = {
    "base_url": "https://huggingface.co/utter-project/mHuBERT-147/resolve/main",
    "files": ("model.safetensors", "config.json", "preprocessor_config.json"),
}
def embedder_directory(embedder):
    """Resolve an embedder name (or a direct directory path) to its model dir."""
    if not embedder or embedder == "hubert_base":
        return HUBERT_MODEL_PATH
    candidate = Path(embedder).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    canonical_name = EMBEDDER_ALIASES.get(embedder, embedder)
    return (PROJECT_ROOT / "assets" / canonical_name).resolve()


def _download_embedder(model_dir, embedder_name=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    embedder_name = embedder_name or model_dir.name
    spec = EMBEDDER_DOWNLOAD_SPECS.get(embedder_name)
    if spec is None:
        raise ValueError("No download source registered for embedder: %s" % embedder_name)
    for file_name in spec["files"]:
        target = model_dir / file_name
        if target.is_file():
            continue
        url = "%s/%s" % (spec["base_url"], file_name)
        logger.info("Downloading %s embedder file %s", embedder_name, url)
        temp_path = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url) as response, open(temp_path, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(temp_path, target)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise


def resolve_embedder_path(embedder):
    """Return the embedder model directory, downloading weights when needed."""
    model_dir = embedder_directory(embedder)
    canonical_name = EMBEDDER_ALIASES.get(embedder, embedder)
    spec = EMBEDDER_DOWNLOAD_SPECS.get(canonical_name)
    if spec is not None:
        if any(not (model_dir / name).is_file() for name in spec["files"]):
            _download_embedder(model_dir, canonical_name)
    elif not (model_dir / "config.json").is_file():
        if embedder not in EMBEDDER_CHOICES or embedder == "hubert_base":
            raise FileNotFoundError(
                "Embedder model not found: %s (expected config.json)" % model_dir
            )
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("Embedder model not found: %s" % model_dir)
    return model_dir


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def load_hubert_model(device, is_half=False, embedder="hubert_base"):
    """Load the local Transformers HuBERT/ContentVec/SPIN embedder for NVC."""
    model_path = resolve_embedder_path(embedder)

    dtype = torch.float16 if is_half else torch.float32
    load_options = {
        "local_files_only": True,
        "torch_dtype": dtype,
    }
    # DirectML does not implement every SDPA kernel used by Transformers.
    if _device_type(device) == "privateuseone":
        load_options["attn_implementation"] = "eager"

    logger.info(
        "Loading Transformers embedder %s from %s (%s on %s)",
        embedder,
        model_path,
        dtype,
        device,
    )
    model = HubertModel.from_pretrained(str(model_path), **load_options)
    model._nvc_embedder_name = embedder
    model._nvc_layer_mix = (
        PHONETIC_LAYER_MIX if embedder == "mhubert147-phonetic" else None
    )
    model._nvc_audio_requires_normalization = hubert_audio_requires_normalization(
        embedder
    )
    model = model.to(device)
    return model.eval()


@lru_cache(maxsize=8)
def hubert_audio_requires_normalization(embedder="hubert_base"):
    try:
        model_path = resolve_embedder_path(embedder)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(model_path), local_files_only=True
        )
        return bool(feature_extractor.do_normalize)
    except Exception:
        # Embedders without a preprocessor config (e.g. ContentVec) are fed
        # raw audio, mirroring Applio behavior.
        logger.debug("No feature extractor config for embedder %s", embedder)
        return False


def extract_hubert_features(model, source, padding_mask=None):
    """Return the selected 768-D HuBERT representation used by NVC v2."""
    if getattr(model, "_nvc_audio_requires_normalization", False):
        source = F.layer_norm(source, source.shape[-1:])

    attention_mask = None
    if padding_mask is not None and bool(torch.any(padding_mask).item()):
        attention_mask = (~padding_mask.bool()).long()

    layer_mix = getattr(model, "_nvc_layer_mix", None)

    if layer_mix:
        graph_key = "hubert-phonetic-mask" if attention_mask is not None else "hubert-phonetic-no-mask"

        def phonetic_forward(input_values, mask=None):
            outputs = model(
                input_values=input_values,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            return sum(hidden_states[layer] * weight for layer, weight in layer_mix)

        if attention_mask is None:
            return run_cuda_graph(model, graph_key, phonetic_forward, source)
        return run_cuda_graph(
            model, graph_key, phonetic_forward, source, attention_mask
        )

    if attention_mask is None:
        def forward(input_values):
            return model(
                input_values=input_values,
                attention_mask=None,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state

        return run_cuda_graph(model, "hubert-no-mask", forward, source)

    def forward(input_values, mask):
        return model(
            input_values=input_values,
            attention_mask=mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

    return run_cuda_graph(model, "hubert-mask", forward, source, attention_mask)
