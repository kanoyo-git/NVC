import logging
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from transformers import AutoFeatureExtractor, HubertModel

from tools.cuda_graph import run_cuda_graph


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

HUBERT_MODEL_PATH = (PROJECT_ROOT / "assets" / "hubert_base").resolve()

# Embedder registry: name -> directory under assets/. The default hubert_base
# ships with NVC; the other embedders are downloaded on first use (weights are
# MIT-licensed and mirrored from the Applio project).
EMBEDDER_CHOICES = ("hubert_base", "contentvec", "spin", "spin-v2")
EMBEDDER_DOWNLOAD_FILES = ("pytorch_model.bin", "config.json")
EMBEDDER_DOWNLOAD_URL = (
    "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders"
)


def embedder_directory(embedder):
    """Resolve an embedder name (or a direct directory path) to its model dir."""
    if not embedder or embedder == "hubert_base":
        return HUBERT_MODEL_PATH
    candidate = Path(embedder).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (PROJECT_ROOT / "assets" / embedder).resolve()


def _download_embedder(model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    embedder_name = model_dir.name
    for file_name in EMBEDDER_DOWNLOAD_FILES:
        target = model_dir / file_name
        if target.is_file():
            continue
        url = "%s/%s/%s" % (EMBEDDER_DOWNLOAD_URL, embedder_name, file_name)
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
    if not (model_dir / "config.json").is_file():
        if embedder not in EMBEDDER_CHOICES or embedder == "hubert_base":
            raise FileNotFoundError(
                "Embedder model not found: %s (expected config.json)" % model_dir
            )
        _download_embedder(model_dir)
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
    model = HubertModelWithFinalProj.from_pretrained(
        str(model_path), **load_options
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


def extract_hubert_features(model, source, version, padding_mask=None):
    """Return the NVC v1 (256-D) or v2 (768-D) HuBERT representation.

    Transformers hidden_states[N] is numerically equivalent to the source checkpoint's
    output_layer=N for this converted checkpoint. NVC v1 uses layer 9 followed
    by final_proj; NVC v2 uses the final (12th) encoder layer directly.
    """
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported NVC feature version: {version!r}")

    attention_mask = None
    if padding_mask is not None and bool(torch.any(padding_mask).item()):
        attention_mask = (~padding_mask.bool()).long()

    if version == "v1":
        if attention_mask is None:
            def forward(input_values):
                outputs = model(
                    input_values=input_values,
                    attention_mask=None,
                    output_hidden_states=True,
                    return_dict=True,
                )
                return model.final_proj(outputs.hidden_states[9])

            return run_cuda_graph(model, "hubert-v1-no-mask", forward, source)

        def forward(input_values, mask):
            outputs = model(
                input_values=input_values,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return model.final_proj(outputs.hidden_states[9])

        return run_cuda_graph(
            model, "hubert-v1-mask", forward, source, attention_mask
        )

    if attention_mask is None:
        def forward(input_values):
            return model(
                input_values=input_values,
                attention_mask=None,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state

        return run_cuda_graph(model, "hubert-v2-no-mask", forward, source)

    def forward(input_values, mask):
        return model(
            input_values=input_values,
            attention_mask=mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

    return run_cuda_graph(model, "hubert-v2-mask", forward, source, attention_mask)
