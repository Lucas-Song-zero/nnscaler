# Qwen3.5-0.8B on nnScaler: Exploration Plan

**Revised:** 2026-11-16

## 1. Scope and important corrections

Investigate **text-only, full-parameter causal-LM training** first. The questions are:

1. Can nnScaler represent both Gated DeltaNet (GDN) and gated full attention correctly in FX/IR?
2. Which partitions are legal, and which does AutoDist actually choose?
3. Does the resulting program improve throughput or memory over an equivalent baseline?

This is an experimental integration plan, **not a claim of existing Qwen3.5 support**.
No Qwen training, model download, or package installation was performed when writing it.
Commands below are intended for your terminal. GPU commands require an allocation:
this machine may be shared, so do not assume all four devices are free.

Corrections to the initial plan:

- Use the **official** `Qwen/Qwen3.5-0.8B` checkpoint. Transformers provides
  `Qwen3_5ForCausalLM` for its text backbone; a community text-only conversion is not required.
- Do not use the generic `examples/huggingface_nlp/compile_hf.py --reduce` as the main entry.
  Its generic reduction does not preserve the nested Qwen text configuration and hybrid
  layer schedule. Its compiler also hardcodes `ComputeConfig(1, 1)` and a DP policy.
- `attn_implementation="eager"` controls full attention, **not the GDN backend**.
  GDN may still use FLA or causal-convolution extensions if installed.
- A successful forward pass or descending loss does not prove correct distributed
  gradients. Numerical parity is a separate requirement.
- `plan_ngpus=4, runtime_ngpus=4` removes outer DP replication, but does not force
  every operation or weight to be sharded. Batch partitioning and replication can
  still occur inside the plan.

Multimodal inputs, autoregressive cache/decode, MTP training, LoRA, quantization,
and GDN sequence-parallel algorithms are deferred.

## 2. Architecture to preserve

The official checkpoint's `text_config` specifies:

| Component | Qwen3.5-0.8B text configuration |
|---|---|
| Decoder layers | 24: 18 GDN and 6 full-attention layers |
| Schedule | `[GDN, GDN, GDN, full attention]` repeated six times |
| Hidden size / FFN intermediate size | 1024 / 3584 |
| Vocabulary | 248320 |
| GDN Q/K heads / V heads | 16 / 16 |
| GDN key/value head dimensions | 128 / 128 |
| GDN causal convolution width | 4 |
| Full-attention Q heads / KV heads | 8 / 2 |
| Full-attention head dimension | 256 |
| Rotary dimension | 64, with the original RoPE configuration retained |
| Output | Tied token embedding and LM head |

This small variant has a **dense FFN**, not an MoE FFN. Family-level marketing
descriptions should not be used to infer its concrete architecture.

```text
Token IDs
  -> embedding [248320, 1024]
  -> 6 x {
       3 x [norm -> GDN -> residual -> norm -> dense gated FFN -> residual]
       1 x [norm -> gated full attention -> residual
                 -> norm -> dense gated FFN -> residual]
     }
  -> final norm
  -> tied LM head
  -> next-token loss
```

In the inspected Transformers v5.3.0 implementation:

- **GDN:** QKV projections -> depthwise causal convolution -> Q/K normalization
  and gated delta-rule computation -> gated RMSNorm -> output projection.
  Separate projections produce the output gate, beta, and decay inputs.
- **Full attention:** joint query/gate projection, K/V projections, Q/K norms,
  RoPE, causal grouped-query attention, then `attention_output * sigmoid(gate)`
  before the output projection.

GDN's recurrent dependence is along sequence, not across independent heads.
That makes head partitioning a sensible first target. Do not label its sequence
axis independently partitionable without a correct state-transfer algorithm.

## 3. Workspace and environment

### 3.1 Preserve the working nanoGPT environment

Run these commands in a shell where your existing conda installation is initialized.
The CLI shell used to write this document did not have `conda` or `nvidia-smi` on
PATH; that does not establish whether they work in your normal terminal.

```bash
conda activate nnscaler
export REPO=/home/v-xuasong/nnscaler
export WORK="$HOME/nnscaler-experiments/qwen3_5_0_8b"
mkdir -p "$WORK"/{models,data,logs,artifacts}
set -o pipefail

python -V
python -m pip freeze > "$WORK/artifacts/nanogpt-env-before.txt"
git -C "$REPO" rev-parse HEAD > "$WORK/artifacts/nnscaler-commit.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
nvidia-smi topo -m

conda create --name nnscaler-qwen35 --clone nnscaler
conda activate nnscaler-qwen35
```

Use the new environment for all subsequent commands. Keep model files, datasets,
generated code, and checkpoints outside the repository. Avoid automatic resume
from nanoGPT or another Qwen experiment's `last` checkpoint.

### 3.2 Dependency compatibility is the first gate

This nnScaler checkout declares `torch>=2.0,<=2.6`. Transformers **v5.3.0**
contains the Qwen3.5 classes and declares `torch>=2.4` for its PyTorch extra.
Thus 2.4--2.6 is a candidate overlap, **not a tested compatibility guarantee**.
The old example pin `transformers==4.40.0` is unsuitable. The checkpoint's
`transformers_version` field is not a minimum-version guarantee either.

Preserve the cloned CUDA-enabled torch build rather than allowing a silent upgrade:

```bash
python - <<'PY'
import os
from pathlib import Path
from packaging.version import Version
import torch

v = Version(torch.__version__.split("+")[0])
print("torch:", torch.__version__, "CUDA build:", torch.version.cuda)
if not Version("2.4") <= v <= Version("2.6"):
    raise SystemExit(
        "Stop: select a CUDA-enabled torch 2.4--2.6 build compatible with "
        "your driver in this cloned environment before proceeding."
    )
Path(os.environ["WORK"], "constraints.txt").write_text(
    f"torch=={torch.__version__}\ntransformers==5.3.0\n"
)
PY
```

If that check stops, use the [official PyTorch version-specific installation
commands](https://pytorch.org/get-started/previous-versions/) for a compatible
CUDA build, then repeat the check. Keep torchvision/torchaudio compatible if
installed. Do not solve conflicts by removing nnScaler's torch bound or using
`--no-deps` to hide incompatibilities.

Review the resolver's proposal before installing:

```bash
python -m pip install --dry-run \
  --report "$WORK/artifacts/qwen-dependency-proposal.json" \
  -c "$WORK/constraints.txt" \
  "transformers==5.3.0" "huggingface_hub>=1.3,<2" safetensors

# Only after reviewing the proposal:
python -m pip install -c "$WORK/constraints.txt" \
  "transformers==5.3.0" "huggingface_hub>=1.3,<2" safetensors
python -m pip check
python -c 'import nnscaler; print(nnscaler.__file__)'
python -c 'from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig; print("Qwen imports OK")'
python -m pip freeze > "$WORK/artifacts/qwen-env.txt"
hf --help
```

Stop on an import or dependency failure; dependency resolution alone does not
validate nnScaler's tracer against a new Transformers release. If the cloned
environment lacks nnScaler, install this checkout with
`python -m pip install -c "$WORK/constraints.txt" -e "$REPO"`.

Do not install FlashAttention, FLA, Apex, or causal-conv1d yet. Inspect whether
the cloned environment already contains them:

```bash
python - <<'PY'
import importlib.util
for name in ("fla", "causal_conv1d", "flash_attn", "apex"):
    print(name, importlib.util.find_spec(name) is not None)
PY
```

An inherited installation can change the selected GDN implementation. Record it.
A pure-PyTorch reference environment and a fast-kernel environment should be
separate variants, with versions and actual function targets recorded for each.

## 4. Download the official Hugging Face model

### 4.1 Resolve and record an immutable revision

The public checkpoint normally does not require authentication. If your account
or network requires it, use interactive `hf auth login`; do not put a token in
this document, command history, or experiment logs.

```bash
export MODEL_ID=Qwen/Qwen3.5-0.8B
export MODEL_DIR="$WORK/models/Qwen3.5-0.8B"

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

info = HfApi().model_info(os.environ["MODEL_ID"], files_metadata=True)
out = Path(os.environ["WORK"], "artifacts")
(out / "model-revision.txt").write_text(info.sha + "\n")
print("Model:", info.id, "revision:", info.sha)
for f in info.siblings:
    if f.rfilename.endswith(".safetensors"):
        print(f.rfilename, f.size, "bytes")
PY

export MODEL_REVISION="$(cat "$WORK/artifacts/model-revision.txt")"
```

Keep this revision when repeating experiments. `main` can change.

### 4.2 Download configuration first, then weights

```bash
hf download "$MODEL_ID" config.json \
  --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"

python -m json.tool "$MODEL_DIR/config.json"

# Preview download size; reserve room for weights, local exports, and checkpoints.
hf download "$MODEL_ID" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR" --dry-run
df -h "$WORK"

# Downloads the official repository files, including tokenizer and safetensors.
hf download "$MODEL_ID" --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"
du -sh "$MODEL_DIR"
```

Rerun the same download command after an interruption; the Hub client reuses
completed files. Do not delete the model directory to retry.

If the `hf` executable is unavailable but the package imports, the equivalent is:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_DIR"],
))
PY
```

### 4.3 Confirm the architecture without using a GPU

```bash
python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

root = Path(os.environ["MODEL_DIR"])
c = json.loads((root / "config.json").read_text())
t = c["text_config"]
print("Top-level architecture:", c["architectures"])
print("Layer counts:", Counter(t["layer_types"]))
for k in ("hidden_size", "intermediate_size", "vocab_size",
          "num_attention_heads", "num_key_value_heads",
          "linear_num_key_heads", "linear_num_value_heads"):
    print(k, t[k])
assert t["layer_types"] == [
    "linear_attention", "linear_attention", "linear_attention", "full_attention"
] * 6
tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
print("Tokenizer length:", len(tok), "BOS:", tok.bos_token_id,
      "EOS:", tok.eos_token_id, "PAD:", tok.pad_token_id)
PY
```

Do not reuse Shakespeare's character IDs or a Llama tokenizer. Do not assume a
BOS token exists: the existing generic tokenizer helper has a BOS assertion that
may be inappropriate here. Prefer unpadded fixed-length blocks initially.

## 5. Native PyTorch forward/backward before nnScaler

**GPU allocation required.** Select one assigned device, not necessarily GPU 0:

```bash
# Example only: replace 0 with a GPU assigned to your job.
export CUDA_VISIBLE_DEVICES=0
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

The following probe loads only the text model from the official local checkpoint.
It uses short input, no KV cache, no `device_map="auto"`, and no HF tensor-parallel
plan. nnScaler, not Transformers/Accelerate, will own distribution later.

```bash
python - <<'PY' 2>&1 | tee "$WORK/logs/native-smoke.log"
import os
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(), "Use a separately validated FP32 reference if BF16 is unsupported"
torch.manual_seed(0)
path = os.environ["MODEL_DIR"]
cfg = AutoConfig.from_pretrained(path, local_files_only=True).text_config
cfg.use_cache = False
model, loading = Qwen3_5ForCausalLM.from_pretrained(
    path, config=cfg, local_files_only=True,
    dtype=torch.bfloat16, attn_implementation="eager",
    output_loading_info=True,
)
print("Loading report:", loading)
assert not loading.get("missing_keys"), "Do not benchmark randomly initialized text weights"
assert not loading.get("mismatched_keys"), "Checkpoint/config mismatch"
assert not loading.get("error_msgs"), "Checkpoint loading error"
# Vision/MTP-only unused weights can be expected; text-weight omissions are not.
assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
model = model.cuda().train()
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
ids = tok(
    "A small experiment studies gated attention and distributed training. " * 8,
    return_tensors="pt", truncation=True, max_length=128,
)["input_ids"].cuda()

for i, layer in enumerate(model.model.layers):
    if hasattr(layer, "linear_attn"):
        gdn = layer.linear_attn
        print("GDN backend", i, gdn.chunk_gated_delta_rule,
              "causal conv:", gdn.causal_conv1d_fn,
              "norm:", type(gdn.norm).__name__)

torch.cuda.reset_peak_memory_stats()
# HF causal-LM loss shifts labels internally: do not pre-shift them here.
out = model(input_ids=ids, labels=ids.clone(), use_cache=False)
assert torch.isfinite(out.loss)
out.loss.backward()
missing, nonfinite = [], []
for name, p in model.named_parameters():
    if p.requires_grad:
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
print("loss:", out.loss.item())
print("missing gradients:", missing, "nonfinite gradients:", nonfinite)
print("peak allocated GiB:", torch.cuda.max_memory_allocated() / 2**30)
assert not missing and not nonfinite
PY
```

This is a forward/backward smoke check, not an optimizer-memory or performance
benchmark. A short sequence still materializes logits over 248320 vocabulary
entries. At 4096 tokens and batch 1, BF16 logits alone are about **1.89 GiB**;
FP32 loss intermediates and gradients add more.

If checkpoint conversion fails, investigate the pinned Transformers loading
mapping. Do not silently substitute random weights or an unrelated model.
The inspected v5.3.0 mapping renames `model.language_model` to `model` for
`qwen3_5_text`. Optionally export a verified text-only copy with
`model.save_pretrained(...)` and `tokenizer.save_pretrained(...)`, but that is
not needed for the initial download or probe.

## 6. Four-layer FX/IR compatibility probe

Start with four layers rather than one: retain all three GDN blocks and the full
attention block. Keep their real head dimensions. This is a **randomly initialized
debug model**, not the pretrained 0.8B model, and retains the large vocabulary.

Run on the assigned single GPU after the native probe. This command deliberately
attempts the unmodified model first; a trace/parser failure is useful evidence,
not an unexpected promise violation.

```bash
export TRACE_DIR="$WORK/artifacts/mini-fx-ir-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TRACE_DIR"

python - <<'PY' 2>&1 | tee "$TRACE_DIR/trace.log"
import os
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph

torch.manual_seed(0)
root = Path(os.environ["TRACE_DIR"])
c = AutoConfig.from_pretrained(
    os.environ["MODEL_DIR"], local_files_only=True
).text_config
c.num_hidden_layers = 4
c.layer_types = c.layer_types[:4]
c.use_cache = False
c._attn_implementation = "eager"

class LossWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen3_5ForCausalLM(c)

    def forward(self, input_ids, labels):
        return self.model(
            input_ids=input_ids, labels=labels, use_cache=False,
            return_dict=False,
        )[0]

m = LossWrapper().to(device="cuda", dtype=torch.bfloat16).train()
ids = torch.randint(0, c.vocab_size, (1, 128), device="cuda")
args = {"input_ids": ids, "labels": ids.clone()}
reference = m(**args).detach()
fx = to_fx_graph(m, args)
(root / "fx.txt").write_text(str(fx.graph))
(root / "fx_code.py").write_text(fx.code)
actual = fx(**args).detach()
print("Reference loss:", reference.item(), "FX loss:", actual.item())
print("FX absolute loss difference:", (reference - actual).abs().item())
print("FX nodes:", Counter(n.op for n in fx.graph.nodes))

attrs = root / "ir_attrs"
attrs.mkdir(exist_ok=True)
ir = to_ir_graph(fx, args, attr_savedir=attrs, constant_folding=True)
(root / "ir.txt").write_text(str(ir))
(root / "ir_nodes.txt").write_text("\n".join(str(n) for n in ir.nodes()))
print("IR node classes:", Counter(type(n).__name__ for n in ir.nodes()))
PY
```

The direct `to_ir_graph` result is the **forward logical graph**, not the final
partitioned forward/backward execution plan. The subsequent `parallelize` pass
must separately capture backward construction, placement, adapters, and codegen.

Preserve the first failing traceback. Classify it as one of: dependency/import,
native kernel, FX trace, IR parsing, partition transformation, profiling, codegen,
distributed runtime, or numerical mismatch. Avoid changing all layers at once.

## 7. Model adaptation and legal partitions

These are implementation tasks, not features already provided by this plan:

| Area | Initial approach | Required evidence |
|---|---|---|
| GDN core | Register a module-level function wrapping the delta-rule kernel | Output and all input-gradient parity |
| GDN convolution | Keep depthwise convolution semantics; adapt group counts if partitioned | Unsplit vs split output/gradient match |
| Gated RMSNorm | Keep gate and normalized-value ordering and dtype behavior | Gate/norm parameter-gradient parity |
| Full attention | Start eager; then use an explicitly registered efficient kernel | Preserve Q/K norm, RoPE, GQA and output gate |
| Embedding/LM head | Preserve tied weights | Single logical weight and correct accumulated gradient |
| Loss | Explicitly choose HF-shifted or externally shifted labels, never both | Same target tokens and global normalization |

For the no-cache GDN core with Q/K/V already shaped `[B, S, H, D]`,
and gates shaped `[B, S, H]`, a **candidate** annotation is:

```text
b s^ h dk^, b s^ h dk^, b s^ h dv^, b s^ h, b s^ h
    -> b s^ h dv^
```

This assumes a wrapper accepting Q, K, V, decay, beta and returning only output,
with zero initial state, no returned final state, and Q/K normalization handled
consistently. Batch and head splits must be validated; sequence and head-feature
axes are deliberately unsplittable. It is not a drop-in annotation for every FLA
API. Register global functions, pass all parameter tensors explicitly, and do not
hide trainable weights inside a registered module or closure.

Test individual GDN/full-attention blocks before the four-layer model. Include
non-chunk-aligned lengths (for example 127, 128, 129), causal behavior, and head
partitions of 2 and 4 where legal. Use FP32 references when supported and compare
BF16 drift against native-kernel drift, rather than picking a tolerance that
hides errors. Record loss, activation, gradient and optimizer-update differences.

**Important asymmetry:** GDN has 16 heads, while full attention has only 2 KV
heads. The current generic grouped-query FlashAttention annotation partitions
whole KV groups, so it only directly offers a two-way head split for that kernel.
A four-GPU plan may need replicated KV, different operator layouts, pipeline
stages, or batch splits. A four-way Q-head split with replicated KV requires its
own correct representation; it is not automatically implied by `plan_ngpus=4`.

Use `input_gen_fn` for valid profiling inputs when necessary: normalized Q/K,
appropriate gate/decay ranges, supported layouts, and realistic dtypes. Profile
forward **and backward** of the registered training kernel. Record actual backend
function names; eager full attention does not guarantee reference GDN.

Do not add fast kernels until the reference path is understood. Then select and
pin FLA/causal-conv1d/FlashAttention versions compatible with the chosen
torch/CUDA/driver stack. If their requirements exceed nnScaler's torch range,
treat that as a separate porting task, not a reason to upgrade the working env.

## 8. Prepare a bounded dataset

For compiler parity, use deterministic synthetic token IDs. For training curves,
use real text tokenized with this checkpoint. The following creates a small
offline smoke dataset from the existing nanoGPT Shakespeare input; it is not a
representative performance or model-quality benchmark.

```bash
export SEQ_LEN=512
python - <<'PY'
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer

src = Path(os.environ["REPO"], "examples/nanogpt/nanoGPT/data/shakespeare_char/input.txt")
assert src.exists(), "Use an existing permitted UTF-8 corpus or prepare nanoGPT data first"
text = src.read_text()
cut = int(len(text) * 0.9)
tok = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"], local_files_only=True)
length = int(os.environ["SEQ_LEN"])
out = Path(os.environ["WORK"], "data", f"shakespeare-qwen-{length}")
out.mkdir(parents=True, exist_ok=True)
for split, part in (("train", text[:cut]), ("validation", text[cut:])):
    ids = tok.encode(part, add_special_tokens=False)
    count = len(ids) // length
    assert count > 0
    blocks = torch.tensor(ids[:count * length], dtype=torch.long).view(count, length)
    torch.save({"input_ids": blocks}, out / f"{split}.pt")
    print(split, tuple(blocks.shape))
(out / "metadata.json").write_text(json.dumps({
    "source": str(src), "model_revision": os.environ["MODEL_REVISION"],
    "sequence_length": length, "labels": "HF internal next-token shift",
}, indent=2))
PY
```

The future trainer must load these tensors explicitly; they are not a
`datasets.load_from_disk` dataset. Split text before tokenization/blocking to
avoid train/validation overlap. Count non-ignored target tokens, including the
loss of one target per block when using internal causal shifting.

## 9. Native Trainer adapter and proposed command interface

Implement a new `examples/qwen3_5/train.py` rather than editing nanoGPT. Reuse
the Native Trainer lifecycle from `examples/longrope2/train.py`, but not its
Llama RoPE patch or assumptions about modern HF attention registry APIs.

The adapter needs:

- Explicit `Qwen3_5ForCausalLM` loading with the text configuration; scalar loss
  first in its output and sample input first for end-to-end mode.
- A tensor-block dataset/collator, deterministic sample ordering, and correct
  per-token gradient normalization. No extra label shift when using HF loss.
- `micro_batch_size`, accumulation, precision, seed, max steps, and checkpoint
  directories exposed as arguments. A baseline native/DDP path with identical
  loss and optimizer behavior.
- `ComputeConfig` wiring for `plan_ngpus`, `runtime_ngpus`, `use_end2end=True`,
  recomputation, memory budget, and explicit pipeline options.
- Distinct generated-code, plan, profile and checkpoint namespaces per experiment.
  Fresh compilation after changing kernel registrations or communication profiles.
- Optional chunked LM-head/loss after proving parity: reusing the generic chunk
  loss may restrict vocab partitioning, so record this search-space tradeoff.

**The commands in this section are proposed interface examples, not runnable
today: `examples/qwen3_5/train.py` does not yet exist.** The preceding download,
native smoke, and FX/IR probe commands do not depend on it.

```bash
# AFTER implementing the adapter and its arguments:
# Choose a per-GPU memory budget from your actual allocated GPUs.
export GPU_MEM_GIB=20  # Example for a 24-GiB card; replace for your hardware.

python "$REPO/examples/qwen3_5/train.py" \
  --run_mode compile --model_path "$MODEL_DIR" \
  --dataset_path "$WORK/data/shakespeare-qwen-512" \
  --plan_ngpus 4 --runtime_ngpus 4 \
  --micro_batch_size 1 --grad_accumulation_steps 8 \
  --gpu_mem_constraint "$GPU_MEM_GIB" \
  --pipeline_nstages 1 --max_train_steps 50 \
  --output_dir "$WORK/artifacts/p4-r4-s512"

# Set only the four devices actually allocated to your job.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  "$REPO/examples/qwen3_5/train.py" \
  --run_mode run --model_path "$MODEL_DIR" \
  --dataset_path "$WORK/data/shakespeare-qwen-512" \
  --plan_ngpus 4 --runtime_ngpus 4 \
  --micro_batch_size 1 --grad_accumulation_steps 8 \
  --gpu_mem_constraint "$GPU_MEM_GIB" \
  --pipeline_nstages 1 --max_train_steps 50 \
  --output_dir "$WORK/artifacts/p4-r4-s512"
```

Initially disable pipeline search. Later enable it explicitly with decoder-layer
pivots, e.g. `Qwen3_5DecoderLayer`, `pipeline_nstages="auto"`,
`use_end2end=True`, and `use_async_reducer=False`. Verify the pivot matches
the actual module names. Use `mem_constraint` inside `pas_config`, rather than
the internal AutoDist field name `memory_constraint`.

Compile-only is not necessarily CPU-only: tracing and operator profiling can
require substantial GPU memory and time.

## 10. Communication profiling and controlled experiment matrix

Run communication profiling only while the allocated GPUs are otherwise idle.
This existing command is runnable once nnScaler and CUDA work:

```bash
# Replace the device list with your allocation.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m nnscaler.profiler.benchmark_comm \
  2>&1 | tee "$WORK/logs/comm-profile.log"

python - <<'PY'
from nnscaler.autodist.util import get_default_profile_path
print("Default AutoDist profile root:", get_default_profile_path())
PY
```

The profiler writes to the default user cache; it can replace existing measurements.
Archive existing relevant comm JSON files first if earlier experiments depend on
them, and archive the new ones with each Qwen experiment. Confirm the compile log
actually loads them. Do not infer use from the fact that profiling completed.

To isolate profile effects, keep model/backend/settings fixed and compare two
**freshly compiled** plans, with distinct artifact directories and no forced
`load_plan_path`. `reuse='override'` can remove a prior generated instance, so use
it only in a dedicated experiment directory. A changed profile can legitimately
select the same plan.

For the initial training comparison, fix global sequences per optimizer update
at 8, microbatch at 1 per plan replica, and sequence length at 512:

| Experiment | Plan GPUs | Runtime GPUs | Outer DP replicas | Accumulation | Global sequences/update |
|---|---:|---:|---:|---:|---:|
| Native reference | - | 1 | 1 | 8 | 8 |
| nnScaler single GPU | 1 | 1 | 1 | 8 | 8 |
| Native DDP reference | - | 4 | 4 | 2 | 8 |
| nnScaler DP | 1 | 4 | 4 | 2 | 8 |
| nnScaler hybrid | 2 | 4 | 2 | 4 | 8 |
| nnScaler single plan | 4 | 4 | 1 | 8 | 8 |

For this sampler contract:

```text
outer_DP = runtime_ngpus / plan_ngpus
global_sequences_per_update = micro_batch_size * accumulation * outer_DP
```

Verify actual sample IDs and unique target-token counts; do not multiply by
runtime ranks when ranks are processing partitions of the same examples.
Hold initial weights, optimizer, LR schedule, clipping, token budget, loss
normalization and evaluation set fixed. Use the same backend across native and
nnScaler comparisons. Keep ZeRO/recomputation explicit and separate their effects.

Progress from 512 to 2048 and 4096 tokens only after correctness. Start with
roughly 10 warmup and 50 measured updates, repeated three times; increase the
window if variance remains large. Do not start at the advertised 262K context.
Use a representative, permitted text corpus for meaningful quality evaluation.

## 11. Evidence, measurements, and exit criteria

Save these separately for every model/backend/shape/plan combination:

| Artifact | What it establishes |
|---|---|
| Environment freeze, repo commit, HF revision | Reproducible inputs |
| Architecture and parameter names/shapes | Actual model loaded; tied weights |
| FX graph and generated FX Python | Traced path, kernel targets, gates retained |
| Forward logical IR and op annotations | Represented operations and legal partitions |
| AutoDist saved plan and compile logs | Selected placement, predicted costs, profile source |
| Post-partition IR/adapters and `gencode*.py` | Actual communication and execution |
| Parameter full maps per rank | Replicated vs partitioned weights, not just shard-file counts |
| Runtime profiler and memory peaks | Measured bottlenecks, not cost-model estimates |
| Loss/gradient/update parity report | Numerical correctness |

FX and pre-partition IR need not change when only `plan_ngpus` changes. Inspect
the transformed graph, adapters and rank code to see placement differences.
Node counts alone are insufficient, especially when a GDN kernel is one opaque op.
Do not present a schematic graph as a captured execution artifact.

Measure steady-state unique target tokens/sec, optimizer-step latency, peak
allocated/reserved memory on every rank, compile/profile/search time, and
communication time by primitive. Synchronize GPUs at timing-window boundaries
and report the slowest-rank elapsed time; separate data loading, compilation,
checkpointing and validation from the core timing window.

```text
speedup = four_GPU_tokens_per_second / one_GPU_tokens_per_second
parallel_efficiency = speedup / 4
```

If the single-GPU setting OOMs, report a capacity improvement, not a fabricated
speedup. Avoid a simple dense-Transformer `6 * parameters * tokens` MFU claim:
GDN and full-attention kernels need an appropriate operation-level FLOP model.

**Milestones:**

1. Native checkpoint loading and finite forward/backward.
2. Isolated GDN and full-attention numerical/gradient parity.
3. Four-layer FX/IR conversion with both attention types preserved.
4. Single-GPU nnScaler parity, including at least one optimizer update.
5. Two-/four-GPU parity after gathering logical gradients/weights correctly.
6. Full 24-layer training, checkpoint merge/reload parity, then performance sweeps.

Report compiler compatibility, partition support, and efficiency as separate
outcomes. Pure replication can be a rational choice for a small workload; forcing
sharding merely to obtain different rank files is not success. Conversely, an
opaque replicated GDN core does not establish efficient GDN parallelization.

## 12. Sources and implementation references

External sources inspected for this revision:

- [Official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Official configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
- [HF download/authentication CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Current Qwen3.5 class documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Pinned v5.3.0 modeling code](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Pinned v5.3.0 checkpoint conversion](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/conversion_mapping.py)
- [Pinned v5.3.0 dependency declarations](https://github.com/huggingface/transformers/blob/v5.3.0/setup.py)
- [FLA installation and implementation](https://github.com/fla-org/flash-linear-attention)
- [Causal convolution implementation](https://github.com/Dao-AILab/causal-conv1d)

Relevant paths relative to this repository root:

- `requirements.txt`: torch compatibility bound.
- `examples/longrope2/train.py`: Native Trainer compile/run structure.
- `examples/transformers_utils/causal_lm_wrapper.py`: loss wrapper pattern.
- `examples/transformers_utils/flash_attn_anno.py`: GQA partition annotation.
- `examples/transformers_utils/chunk_linear_cross_entropy.py`: large-vocabulary loss.
- `examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py`:
  registered kernels and profiling-input pattern, not a Qwen implementation.
- `docs/source/register_custom_op.md`: custom-op partition contracts.
- `docs/source/control_flow.md`: concrete-trace branch limitations.
- `nnscaler/graph/parser/converter.py`: FX and logical IR conversion.
- `nnscaler/policies.py`: actual AutoDist/PP configuration gates.
- `nnscaler/profiler/benchmark_comm.py`: communication profiling.

