"""Catalog of community RVC v2 pretrained G/D pairs.

The catalog lives in configs/pretrains.json (name -> sample rate -> G/D paths on
huggingface.co). Downloaded pairs are validated against the NVC v2 architecture,
stripped of optimizer state and stored as
assets/pretrained_v2/<slug>/f0G<sr>.pth + f0D<sr>.pth, which is the layout the
training tab already scans. The stock RVC v2 pair keeps its historical location
directly under assets/pretrained_v2.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "configs" / "pretrains.json"
PRETRAINED_ROOT = Path("assets") / "pretrained_v2"
HF_BASE = "https://huggingface.co/"
DEFAULT_NAME = "RVC v2"
SAMPLE_RATES = ("32k", "40k", "48k")


def load_catalog():
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "pretrain"


def target_dir(name):
    if name == DEFAULT_NAME:
        return PRETRAINED_ROOT
    return PRETRAINED_ROOT / slug(name)


def target_paths(name, sr):
    folder = target_dir(name)
    return folder / ("f0G%s.pth" % sr), folder / ("f0D%s.pth" % sr)


def is_installed(name, sr):
    return all(path.is_file() for path in target_paths(name, sr))


def catalog_entries(sr=None):
    entries = []
    for name, rates in load_catalog().items():
        available = [rate for rate in SAMPLE_RATES if rate in rates]
        if sr and sr not in available:
            continue
        generator, discriminator = target_paths(name, sr) if sr else (None, None)
        entries.append(
            {
                "name": name,
                "sample_rates": available,
                "installed": [rate for rate in available if is_installed(name, rate)],
                "pretrained_g": generator.as_posix() if generator else "",
                "pretrained_d": discriminator.as_posix() if discriminator else "",
            }
        )
    return entries


def _download(url, dest, on_progress=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = Request(url, headers={"User-Agent": "NVC-Studio/1.0"})
    with urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if on_progress:
                on_progress(done, total)
    return partial


def _reference_state(role, sr):
    from infer.module.models import (
        MultiPeriodDiscriminator,
        SynthesizerTrnMs768NSFsid,
    )

    if role == "D":
        return MultiPeriodDiscriminator(False).state_dict()
    with (PROJECT_ROOT / "configs" / "v2" / ("%s.json" % sr)).open(encoding="utf-8") as handle:
        config = json.load(handle)
    model = SynthesizerTrnMs768NSFsid(
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        **config["model"],
        is_half=False,
        sr=sr,
    )
    return model.state_dict()


def validate_and_strip(path, role, sr, dest):
    """Check a downloaded G/D checkpoint against NVC and keep only weights."""
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("%s: no 'model' weights in checkpoint" % path.name)
    reference = _reference_state(role, sr)
    missing = [key for key in reference if key not in state]
    mismatched = [
        key
        for key, value in reference.items()
        if key in state
        and key != "emb_g.weight"
        and tuple(state[key].shape) != tuple(value.shape)
    ]
    if missing or mismatched:
        raise ValueError(
            "%s is not compatible with NVC v2 %s (%s missing, %s shape mismatches%s)"
            % (
                path.name,
                sr,
                len(missing),
                len(mismatched),
                ": " + ", ".join((missing + mismatched)[:4]),
            )
        )
    torch.save(
        {
            "model": {key: state[key] for key in reference},
            "iteration": checkpoint.get("iteration", 0),
            "learning_rate": checkpoint.get("learning_rate", 1e-4),
        },
        str(dest),
    )


def download_pretrained(name, sr, on_progress=None, force=False):
    """Download, validate and install one catalog pair. Returns (G, D) paths."""
    catalog = load_catalog()
    if name not in catalog:
        raise ValueError("Unknown pretrained set: %s" % name)
    if sr not in catalog[name]:
        raise ValueError("%s has no %s pretrained pair" % (name, sr))
    generator, discriminator = target_paths(name, sr)
    if not force and generator.is_file() and discriminator.is_file():
        return generator.as_posix(), discriminator.as_posix()
    files = catalog[name][sr]
    roles = [("G", generator), ("D", discriminator)]
    for index, (role, dest) in enumerate(roles):
        if dest.is_file() and not force:
            continue

        def report(done, total, role=role, index=index):
            if on_progress:
                on_progress(role, index, len(roles), done, total)

        partial = _download(HF_BASE + files[role], dest, report)
        try:
            validate_and_strip(partial, role, sr, dest)
        finally:
            partial.unlink(missing_ok=True)
    return generator.as_posix(), discriminator.as_posix()


def _cli_progress(role, index, count, done, total):
    if total:
        sys.stdout.write("\r%s (%d/%d): %5.1f%%" % (role, index + 1, count, done * 100.0 / total))
    else:
        sys.stdout.write("\r%s (%d/%d): %.1f MB" % (role, index + 1, count, done / 1e6))
    if total and done >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="NVC pretrained catalog")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show catalog entries and installed pairs")
    get = sub.add_parser("download", help="download one pretrained pair")
    get.add_argument("name")
    get.add_argument("sr", choices=SAMPLE_RATES)
    get.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        for entry in catalog_entries():
            installed = ",".join(entry["installed"]) or "-"
            print("%-18s %-14s installed: %s" % (entry["name"], ",".join(entry["sample_rates"]), installed))
        return 0
    generator, discriminator = download_pretrained(
        args.name, args.sr, _cli_progress, force=args.force
    )
    print("G:", generator)
    print("D:", discriminator)
    return 0


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.exit(main())
