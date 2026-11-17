"""Download versioned Transformers Qwen source for inspection, not standalone use."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


VERSION = "5.20.2"
MODEL_PATH = "src/transformers/models/qwen3_5"
FILES = {
    name: f"{MODEL_PATH}/{name}"
    for name in (
        "__init__.py",
        "configuration_qwen3_5.py",
        "modeling_qwen3_5.py",
        "modular_qwen3_5.py",
    )
}
FILES["LICENSE"] = "LICENSE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=VERSION)
    args = parser.parse_args()
    if not all(part.isdecimal() for part in args.version.split(".")):
        parser.error("--version must be a numeric dotted release, e.g. 5.20.2")
    root = Path(__file__).resolve().parent / "source_reference" / f"v{args.version}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "huggingface/transformers", "tag": f"v{args.version}", "files": {}}
    for name, remote_path in FILES.items():
        url = f"https://raw.githubusercontent.com/huggingface/transformers/v{args.version}/{remote_path}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=name)
        (root / name).write_bytes(data)
        manifest["files"][name] = {
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        print(f"Downloaded {name}: {len(data)} bytes", flush=True)
    (root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Reference source: {root}")
    print("Runtime imports still use the installed transformers package.")


if __name__ == "__main__":
    main()
