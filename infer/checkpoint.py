import torch


def load_inference_checkpoint(path, map_location="cpu"):
    """Load and validate the tensor-only checkpoint format used by NVC inference."""
    checkpoint = torch.load(
        path, map_location=map_location, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("NVC checkpoint must be a dictionary")

    config = checkpoint.get("config")
    if not isinstance(config, list) or len(config) < 18:
        raise ValueError("NVC checkpoint is missing a valid config list")

    weights = checkpoint.get("weight")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("NVC checkpoint is missing its weight dictionary")
    if any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in weights.items()
    ):
        raise ValueError("NVC checkpoint weights must contain only named tensors")
    embedding = weights.get("emb_g.weight")
    if not torch.is_tensor(embedding) or embedding.ndim != 2 or embedding.shape[0] < 1:
        raise ValueError("NVC checkpoint has an invalid speaker embedding")
    gin_channels = config[-2]
    if (
        not isinstance(gin_channels, int)
        or isinstance(gin_channels, bool)
        or gin_channels <= 0
        or embedding.shape[1] != gin_channels
    ):
        raise ValueError("NVC checkpoint speaker embedding does not match its config")

    version = checkpoint.get("version", "v1")
    if not isinstance(version, str) or version not in {"v1", "v2"}:
        raise ValueError("Unsupported NVC checkpoint version: %r" % version)
    if_f0 = checkpoint.get("f0", 1)
    if not isinstance(if_f0, (bool, int)) or if_f0 not in {0, 1, False, True}:
        raise ValueError("NVC checkpoint f0 flag must be 0 or 1")
    sample_rate = config[-1]
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
        raise ValueError("NVC checkpoint has an invalid sample rate")
    return checkpoint
