# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Probe native forward, nnScaler FX tracing, and forward IR conversion."""

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import traceback

import torch
from transformers import AutoTokenizer

from model import DEFAULT_MODEL_PATH, QwenTrainingModel


class Tee:
    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--layers", type=int, choices=(4, 8, 12, 16, 20, 24), default=4)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text", default="A small experiment studies gated attention and distributed training.")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are not overwritten")
    parser.add_argument("--native-only", action="store_true", help="Stop after the reduced model's native forward")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for native vs FX loss")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for native vs FX loss")
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if not 0 <= args.atol < float("inf") or not 0 <= args.rtol < float("inf"):
        parser.error("Loss tolerances must be finite and nonnegative")
    return args


def run_probe(args, out, report):
    report["phase"] = "environment"
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for a diagnostic probe")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support BF16; use --dtype float32")
        report["gpu"] = torch.cuda.get_device_name(0)
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers")
    }
    report["cuda_build"] = torch.version.cuda
    report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    report["phase"] = "load_model"
    print("Loading official text weights on CPU...", flush=True)
    wrapper = QwenTrainingModel(args.model_path, dtype=getattr(torch, args.dtype))
    backbone = wrapper.model.model
    if args.layers > len(backbone.layers):
        raise ValueError("Requested more layers than the checkpoint contains")
    backbone.layers = torch.nn.ModuleList(list(backbone.layers[:args.layers]))
    config = wrapper.model.config
    config.num_hidden_layers = args.layers
    config.layer_types = config.layer_types[:args.layers]
    backbone.config.num_hidden_layers = args.layers
    backbone.config.layer_types = list(config.layer_types)
    wrapper = wrapper.to(args.device).train()
    report["layer_types"] = list(config.layer_types)
    report["parameters"] = sum(p.numel() for p in wrapper.parameters())
    report["loading_info"] = {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in wrapper.loading_info.items()
    }
    print("Layer types:", report["layer_types"], flush=True)
    print("Retained parameters:", report["parameters"], flush=True)
    config.save_pretrained(out / "debug_config")

    report["phase"] = "prepare_input"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    ids = tokenizer(
        args.text, return_tensors="pt", truncation=True, max_length=args.max_length,
    )["input_ids"].to(args.device)
    if ids.shape[1] < 2:
        raise ValueError("The text must produce at least two tokens")
    samples = {"input_ids": ids, "labels": ids.clone()}
    dummy_args = {"samples": samples}
    report["input_shape"] = list(ids.shape)
    report["input_ids"] = ids.cpu().tolist()

    report["phase"] = "native_forward"
    with torch.no_grad():
        reference = wrapper(samples).detach()
    if reference.ndim != 0 or not torch.isfinite(reference).item():
        raise RuntimeError("Native forward did not produce a finite scalar loss")
    report["native_loss"] = reference.item()
    print("Native loss:", report["native_loss"], flush=True)
    if args.native_only:
        report.update(status="passed", phase="native_only_complete")
        return

    report["phase"] = "import_nnscaler"
    import nnscaler
    from nnscaler.flags import CompileFlag
    from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph

    report["nnscaler_path"] = nnscaler.__file__
    report["phase"] = "fx_trace"
    report["trace_strategy"] = args.device
    print("Tracing FX (training mode, gradients enabled)...", flush=True)
    previous_strategy = CompileFlag.trace_strategy
    try:
        CompileFlag.trace_strategy = args.device
        fx = to_fx_graph(wrapper, dummy_args)
    finally:
        CompileFlag.trace_strategy = previous_strategy
    (out / "fx_graph.txt").write_text(str(fx.graph), encoding="utf-8")
    (out / "fx_code.py").write_text(fx.code, encoding="utf-8")
    report["fx_node_counts"] = dict(Counter(n.op for n in fx.graph.nodes))

    report["phase"] = "fx_forward_parity"
    with torch.no_grad():
        traced_loss = fx(**dummy_args).detach()
    if traced_loss.ndim != 0 or not torch.isfinite(traced_loss).item():
        raise RuntimeError("FX forward did not produce a finite scalar loss")
    report["fx_loss"] = traced_loss.item()
    report["loss_absolute_difference"] = (reference - traced_loss).abs().item()
    print("FX loss:", report["fx_loss"], flush=True)
    print("Absolute loss difference:", report["loss_absolute_difference"], flush=True)
    torch.testing.assert_close(traced_loss, reference, atol=args.atol, rtol=args.rtol)

    report["phase"] = "ir_conversion"
    print("Converting FX to forward logical IR...", flush=True)
    attrs = out / "ir_attrs"
    attrs.mkdir()
    ir = to_ir_graph(fx, dummy_args, attr_savedir=attrs, constant_folding=True)
    (out / "ir_graph.txt").write_text(str(ir), encoding="utf-8")
    (out / "ir_nodes.txt").write_text(
        "\n".join(str(node) for node in ir.nodes()) + "\n", encoding="utf-8",
    )
    report["ir_node_counts"] = dict(Counter(type(n).__name__ for n in ir.nodes()))
    report.update(status="passed", phase="forward_ir_complete")


def main():
    args = parse_args()
    out = args.output_dir or (
        Path(__file__).resolve().parents[1] / "artifacts"
        / f"probe-{datetime.now():%Y%m%d-%H%M%S-%f}"
    )
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "phase": "starting",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(out),
    }
    print("Artifacts:", out, flush=True)
    handler = logging.FileHandler(out / "python_logging.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        with (out / "probe.log").open("w", encoding="utf-8") as logfile:
            with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
                try:
                    run_probe(args, out, report)
                except Exception as error:
                    report.update(status="failed", error=f"{type(error).__name__}: {error}")
                    detail = traceback.format_exc()
                    (out / "traceback.txt").write_text(detail, encoding="utf-8")
                    print(f"FAILED during {report['phase']}\n{detail}", file=sys.stderr)
                finally:
                    (out / "summary.json").write_text(
                        json.dumps(report, indent=2) + "\n", encoding="utf-8",
                    )
        print(f"Status: {report['status']}; phase: {report['phase']}")
        print("No optimizer update, AutoDist search, or distributed compilation was performed.")
        return 0 if report["status"] == "passed" else 1
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    sys.exit(main())
