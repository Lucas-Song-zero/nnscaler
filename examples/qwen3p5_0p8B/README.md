# Qwen3.5-0.8B: local checkpoint preparation

This directory prepares the **official** `Qwen/Qwen3.5-0.8B` checkpoint for
text-only nnScaler investigation. It is not yet a working nnScaler trainer.
See the [full exploration plan](../qwen3_5_0_8b_nnscaler_exploration_plan.md).

## Files

```text
download_model.py          Revision-pinned Hugging Face download
inspect_model.py           CPU-only configuration and file checks
requirements-download.txt  Downloader dependency only
download_source.py         Download upstream model source matching a release
source_reference/          Ignored by Git; inspection copies, not runtime imports
model/                    Ignored by Git; official config/tokenizer/weights
  download_manifest.json  Resolved revision, sizes, and local SHA-256 hashes
```

Pinned model revision: `2fc06364715b967f1860aea9cf38778875588b17`.
The download includes the safetensors shard and index, config, tokenizer assets,
chat template, model card, and license. It is approximately 1.65 GiB in total.
Image/video preprocessing configs are intentionally omitted for this text-only
stage. The official weight shard still contains vision weights.

## Download and verify

```bash
conda activate nnscaler
cd /home/v-xuasong/nnscaler/examples/qwen3p5_0p8B

# Only if huggingface_hub is missing:
# python -m pip install -r requirements-download.txt

python download_model.py
python inspect_model.py --verify-checksums
```

If conda activation is unavailable in the current shell, the existing interpreter is:

```bash
/home/v-xuasong/miniforge3/envs/nnscaler/bin/python download_model.py
/home/v-xuasong/miniforge3/envs/nnscaler/bin/python inspect_model.py --verify-checksums
```

The public checkpoint normally needs no login. If authentication is required,
use `hf auth login` interactively; never write a token into source or logs.
Rerun the same download command after interruption to reuse completed downloads.
Do not delete the directory to retry.

Checksums in the manifest detect subsequent local changes; they are generated
locally, not a separately signed publisher manifest. Downloads are fetched via
the Hugging Face client at the recorded immutable revision.

## Where is model.py?

The official checkpoint does not need its own standalone `model.py`.
The implementation ships in the **Transformers Python package**:

| File | Purpose |
|---|---|
| `modeling_qwen3_5.py` | Model classes, GDN, gated full attention, and forward methods |
| `configuration_qwen3_5.py` | Text/vision configuration classes |
| `modular_qwen3_5.py` | Upstream source used to generate the modeling implementation |
| `__init__.py` | Package exports |

On 2026-11-17, the existing `nnscaler` environment was inspected and contained
**torch 2.6.0 and Transformers 5.20.2**, with the Qwen3.5 modeling module present.
The installed implementation is at:

```text
/home/v-xuasong/miniforge3/envs/nnscaler/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py
```

Download reference copies matching that release, including its Apache-2.0 license:

```bash
python download_source.py
# Or explicitly select another release for a separate experiment:
# python download_source.py --version 5.3.0
```

The default writes to `source_reference/v5.20.2/` and records download URLs and
SHA-256 hashes in `source_manifest.json`. These files are **for reading**, not a
standalone Python package. They import Transformers utilities and other model
modules, so copying just `modeling_qwen3_5.py` into a training script is insufficient.
`modular_qwen3_5.py` is useful for upstream development but is not required for
ordinary model loading.

Runtime code should continue importing `Qwen3_5ForCausalLM` from `transformers`.
Editing these reference copies will not change the loaded model. Future nnScaler
adaptations should use a dedicated adapter rather than modifying site-packages.
Model loading, forward/backward, and nnScaler compatibility remain separate checks;
finding the implementation does not prove that training works.

## Text-only loading (next stage)

The checkpoint contains a text backbone and vision encoder. Loading it with
`Qwen3_5ForCausalLM` instantiates only the text model, preserving all 24 language
layers: 18 Gated DeltaNet layers and 6 gated full-attention layers.
No community conversion is needed.

This snippet requires a compatible Transformers installation; downloading does
not. Use a separate environment for dependency changes, as described in the plan.
The exploration plan proposed Transformers 5.3.0, while the actual environment
currently has 5.20.2. Do not mix source references from one release with assumptions
about another, or downgrade the working environment automatically. Compatibility
with nnScaler still needs to be established. This checkout bounds torch to
`>=2.0,<=2.6`.

```python
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

path = Path("/home/v-xuasong/nnscaler/examples/qwen3p5_0p8B/model")
config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
    path,
    config=config,
    local_files_only=True,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    output_loading_info=True,
)
print(loading_info)
assert not loading_info.get("missing_keys")
assert not loading_info.get("mismatched_keys")
assert not loading_info.get("error_msgs")
assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
```

Do not ignore missing text weights. Vision/MTP-only unused weights can be expected.
`eager` selects full attention; it does not force the GDN backend to pure PyTorch.
Do not use `device_map="auto"` when preparing nnScaler-managed distribution.

Next milestones are a native forward/backward check, a four-layer hybrid FX/IR
probe, kernel partition annotations, and distributed numerical parity.
Only run GPU work on devices allocated to your job.

## Run the four-layer FX/IR experiment

No new packages or optimized kernels are required for the initial probe.
From your normal terminal:

```bash
conda activate nnscaler
cd /home/v-xuasong/nnscaler/examples/qwen3p5_0p8B/scripts

# Replace 0 with a GPU allocated to your job.
CUDA_VISIBLE_DEVICES=0 python probe.py --device cuda --dtype bfloat16
```

`probe.py` loads the official text weights, keeps the first four decoder layers
(GDN, GDN, GDN, full attention), runs native forward, attempts FX tracing,
compares native/FX scalar loss, and attempts forward logical IR conversion.
The reduced model is for compiler debugging, not quality evaluation; its loss
should not be compared to that of the full 24-layer checkpoint.

Each invocation creates a new timestamped directory under `artifacts/` and prints
its absolute path. No manual `tee` command is needed:

| Output | Contents |
|---|---|
| `probe.log`, `python_logging.log` | Console output and Python logging |
| `summary.json` | Status, phase, settings, versions, input IDs, losses and node counts |
| `traceback.txt` | Exception details if a phase fails |
| `debug_config/` | Configuration of the reduced model |
| `fx_graph.txt`, `fx_code.py` | Written if FX tracing succeeds |
| `ir_graph.txt`, `ir_nodes.txt`, `ir_attrs/` | Written during successful IR conversion |

Exit status 1 means a phase failed; read `summary.json` and `traceback.txt`.
A trace failure is expected diagnostic evidence until Qwen-specific adapters are
implemented. Missing FX/IR files after an earlier failure are normal.
Loss comparison uses `atol=1e-3, rtol=1e-3` by default, configurable via flags.
This is only a scalar forward smoke check, not full activation/gradient parity.

Useful options:

```bash
# CPU native-forward check without invoking the tracer:
python probe.py --device cpu --max-length 8 --native-only

# Later, after the four-layer trace works:
CUDA_VISIBLE_DEVICES=0 python probe.py --layers 24 --max-length 16

python probe.py --help
```

The probe selects the matching CPU/CUDA tracing strategy, but this nnScaler
checkout can still introduce CUDA dependencies during tracing. In the CLI
environment, CPU native forward succeeded while FX tracing reached a CUDA
initialization error. Use your CUDA-enabled terminal for the full experiment;
this is not evidence that the first Qwen kernel is unsupported.

No backward compilation, AutoDist search, optimizer update, or distributed
training is performed by this probe. Keep the fallback kernels and current
environment unchanged while collecting the first traceback.
