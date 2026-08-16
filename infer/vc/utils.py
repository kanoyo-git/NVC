import os
import re

from infer.hubert import load_hubert_model


def _index_candidates(sid, speaker_id=None, include_speaker_specific=False):
    sid_text = os.path.expanduser(str(sid or "")).strip().strip('"')
    model_path = os.path.abspath(sid_text) if os.path.isfile(sid_text) else ""
    model_stem = os.path.splitext(os.path.basename(sid_text))[0]
    experiment_name = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment_name:
        return []

    try:
        target_speaker_id = None if speaker_id is None else int(speaker_id)
    except (TypeError, ValueError):
        target_speaker_id = None
    candidates = []
    roots = [
        os.getenv("outside_index_root"),
        os.getenv("index_root"),
        os.getenv("weight_root"),
        os.path.dirname(model_path) if model_path else None,
    ]
    unique_roots = []
    seen_roots = set()
    for root in roots:
        if not root:
            continue
        absolute_root = os.path.abspath(os.path.expanduser(root))
        if absolute_root not in seen_roots:
            seen_roots.add(absolute_root)
            unique_roots.append(absolute_root)
    preferred_root = unique_roots[0] if unique_roots else ""
    lower_experiment = experiment_name.lower()
    experiment_variants = {
        lower_experiment,
        re.sub(r"_(?:v1|v2)$", "", lower_experiment),
    }
    for index_root in unique_roots:
        if not index_root or not os.path.isdir(index_root):
            continue
        for root, _, files in os.walk(index_root, topdown=False):
            for name in files:
                if not name.lower().endswith(".index") or "trained" in name.lower():
                    continue
                index_stem = os.path.splitext(name)[0]
                lower_index = index_stem.lower()
                lower_experiment = experiment_name.lower()
                speaker_match = re.search(r"_spkid(\d+)$", index_stem, re.IGNORECASE)
                indexed_speaker_id = (
                    int(speaker_match.group(1)) if speaker_match else None
                )
                if (
                    target_speaker_id is None
                    and indexed_speaker_id is not None
                    and not include_speaker_specific
                ):
                    continue
                if (
                    target_speaker_id is not None
                    and indexed_speaker_id is not None
                    and indexed_speaker_id != target_speaker_id
                ):
                    continue
                matching_variant = next(
                    (
                        variant
                        for variant in sorted(experiment_variants, key=len, reverse=True)
                        if variant
                        and re.search(
                            r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(variant),
                            lower_index,
                        )
                    ),
                    "",
                )
                exact_model_match = model_stem.lower() in lower_index
                if matching_variant or exact_model_match:
                    path = os.path.abspath(os.path.join(root, name))
                    score = (
                        0 if indexed_speaker_id == target_speaker_id else 1,
                        0 if matching_variant == lower_experiment else 1,
                        0 if exact_model_match else 1,
                        0 if os.path.abspath(index_root) == preferred_root else 1,
                        -os.path.getmtime(path),
                        path.lower(),
                    )
                    candidates.append((score, path))
    return sorted(candidates, key=lambda item: item[0])


def get_index_paths_from_model(sid):
    """Return all matching index files, ordered by the preferred candidate first."""
    seen = set()
    paths = []
    for _, path in _index_candidates(sid, include_speaker_specific=True):
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def get_index_path_from_model(sid, speaker_id=None):
    candidates = _index_candidates(sid, speaker_id=speaker_id)
    return candidates[0][1] if candidates else ""


def load_hubert(config):
    return load_hubert_model(config.device, config.is_half)
