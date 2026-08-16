import os
import re

from infer.hubert import load_hubert_model


def _index_candidates(sid, speaker_id=None, include_speaker_specific=False):
    model_stem = os.path.splitext(os.path.basename(str(sid or "")))[0]
    experiment_name = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment_name:
        return []

    try:
        target_speaker_id = None if speaker_id is None else int(speaker_id)
    except (TypeError, ValueError):
        target_speaker_id = None
    candidates = []
    roots = [os.getenv("outside_index_root"), os.getenv("index_root")]
    preferred_root = os.path.abspath(roots[0]) if roots[0] else ""
    for index_root in roots:
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
                standard_match = (
                    lower_index.startswith(lower_experiment + "_added_")
                    or ("_" + lower_experiment + "_v1") in lower_index
                    or ("_" + lower_experiment + "_v2") in lower_index
                )
                exact_model_match = model_stem.lower() in lower_index
                if standard_match or exact_model_match:
                    path = os.path.abspath(os.path.join(root, name))
                    score = (
                        0 if indexed_speaker_id == target_speaker_id else 1,
                        0 if standard_match else 1,
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
