"""Download the official checkpoint and record its local file checksums."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00001.safetensors",
    "README.md",
    "LICENSE",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument(
        "--local-dir", type=Path, default=Path(__file__).resolve().parent / "model"
    )
    args = parser.parse_args()
    info = HfApi().model_info(MODEL_ID, revision=args.revision, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    missing = set(FILES) - remote.keys()
    if missing:
        raise RuntimeError(f"Revision has a different file layout: {sorted(missing)}")
    total = sum(remote[name].size for name in FILES)
    print(f"Downloading {MODEL_ID}@{info.sha}: {total / 1024**3:.2f} GiB", flush=True)
    root = args.local_dir.resolve()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=info.sha,
        local_dir=root,
        allow_patterns=FILES,
        max_workers=4,
    )

    manifest = {"model_id": MODEL_ID, "revision": info.sha, "files": {}}
    for name in FILES:
        path = root / name
        size = path.stat().st_size
        if size != remote[name].size:
            raise RuntimeError(f"Size mismatch for {name}: {size} != {remote[name].size}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest["files"][name] = {"size": size, "sha256": digest.hexdigest()}
    (root / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Downloaded and checked {len(FILES)} files in {root}")


if __name__ == "__main__":
    main()
