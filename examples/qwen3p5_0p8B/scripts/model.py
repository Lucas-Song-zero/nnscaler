# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Official Qwen3.5 text weights with a scalar-loss training interface."""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "model"


class QwenTrainingModel(torch.nn.Module):
    """Load text weights without the vision encoder.

    Samples contain input_ids and unshifted labels, both [batch, sequence].
    Labels can contain -100 to ignore targets. Optional attention_mask and
    position_ids are forwarded to the model. HF shifts labels internally.

    The caller owns device placement. No generation cache or HF parallel plan
    is enabled. This wrapper does not yet register nnScaler-specific operators.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        text_config = getattr(config, "text_config", config)
        if not isinstance(text_config, Qwen3_5TextConfig):
            raise ValueError("Expected a Qwen3.5 text or multimodal checkpoint")
        text_config.use_cache = False

        self.model, self.loading_info = Qwen3_5ForCausalLM.from_pretrained(
            model_path,
            config=text_config,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=True,
            output_loading_info=True,
        )
        failures = {
            key: self.loading_info[key]
            for key in ("missing_keys", "mismatched_keys", "error_msgs")
            if self.loading_info.get(key)
        }
        if failures:
            raise RuntimeError(f"Text checkpoint loading failed: {failures}")
        if text_config.tie_word_embeddings and (
            self.model.get_input_embeddings().weight
            is not self.model.get_output_embeddings().weight
        ):
            raise RuntimeError("The checkpoint's tied embeddings were not preserved")
        self.model.train()

    def forward(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return mean next-token loss over non-ignored targets."""
        outputs = self.model(
            input_ids=samples["input_ids"],
            labels=samples["labels"],
            attention_mask=samples.get("attention_mask"),
            position_ids=samples.get("position_ids"),
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--text", default="A small experiment studies gated attention and distributed training.")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--backward", action="store_true", help="Also compute and check gradients; no optimizer step")
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("--max-length must be at least 2 for next-token loss")
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this environment")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            parser.error("The selected CUDA device does not support BF16")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokens = tokenizer(
        args.text, return_tensors="pt", truncation=True, max_length=args.max_length,
    )
    if tokens["input_ids"].shape[1] < 2:
        parser.error("--text must produce at least two tokens")
    samples = {
        "input_ids": tokens["input_ids"].to(args.device),
        "labels": tokens["input_ids"].clone().to(args.device),
        "attention_mask": tokens["attention_mask"].to(args.device),
    }
    wrapper = QwenTrainingModel(
        args.model_path,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    print("Loaded:", args.model_path, flush=True)
    print("Text layers:", wrapper.model.config.num_hidden_layers, flush=True)
    print("Parameters:", sum(p.numel() for p in wrapper.parameters()), flush=True)
    print("Loading report:", wrapper.loading_info, flush=True)
    with torch.set_grad_enabled(args.backward):
        loss = wrapper(samples)
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise RuntimeError(f"Expected finite scalar loss, got {loss}")
    print("Loss:", loss.item(), flush=True)
    if args.backward:
        loss.backward()
        missing, nonfinite = [], []
        for name, parameter in wrapper.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    missing.append(name)
                elif not torch.isfinite(parameter.grad).all().item():
                    nonfinite.append(name)
        if missing or nonfinite:
            raise RuntimeError(f"Missing gradients: {missing}; nonfinite gradients: {nonfinite}")
        print("Backward: all trainable parameters have finite gradients.", flush=True)
    print("Native PyTorch smoke check complete; no nnScaler compilation or optimizer step.")


if __name__ == "__main__":
    main()
