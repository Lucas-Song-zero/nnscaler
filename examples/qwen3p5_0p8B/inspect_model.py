"""Check the downloaded model without loading weights or requiring a GPU."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path(__file__).resolve().parent / "model"
    )
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    root = args.model_dir
    manifest = json.loads((root / "download_manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        path = root / name
        if path.stat().st_size != metadata["size"]:
            raise RuntimeError(f"File size mismatch: {path}")
        if args.verify_checksums:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != metadata["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {path}")

    config = json.loads((root / "config.json").read_text())
    text = config["text_config"]
    expected = ["linear_attention"] * 3 + ["full_attention"]
    if text["layer_types"] != expected * 6:
        raise RuntimeError("Unexpected hybrid layer schedule")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if shard not in manifest["files"]:
            raise RuntimeError(f"Unverified weight shard: {shard}")
    print("Model:", manifest["model_id"])
    print("Revision:", manifest["revision"])
    print("Checkpoint architecture:", config["architectures"])
    print("Text layers:", dict(Counter(text["layer_types"])))
    for name in (
        "hidden_size", "intermediate_size", "vocab_size",
        "num_attention_heads", "num_key_value_heads",
        "linear_num_key_heads", "linear_num_value_heads", "tie_word_embeddings",
    ):
        print(f"{name}: {text[name]}")
    print("Indexed tensors:", len(index["weight_map"]))
    print("Local files verified:", len(manifest["files"]))


if __name__ == "__main__":
    main()
