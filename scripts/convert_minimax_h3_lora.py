"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Convert a MiniMax-H3 or MiniMax-H3-Image CodecOPSD training checkpoint (.pt) into the LoRA file loaded by
`load_condition_lora` (safetensors).

The training checkpoint also stores the full training config; only the LoRA rank / alpha, the token gazer and the VLM
drop are kept. The settings the pipeline implements are checked first, so a checkpoint trained differently fails here
instead of running with the wrong inputs.

    python scripts/convert_minimax_h3_lora.py checkpoint.pt codecopsd_minimax_h3_ref2va.safetensors
"""

import argparse
import re

import torch
from safetensors.torch import save_file

from codecopsd.configs import load

KEY = re.compile(r"transformer\.transformer_blocks\.(\d+)\.(.+)\.lora_(down|up)")
MODULES = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"}
EXPECTED_MODEL = {  # settings implemented by MiniMaxH3CodecOPSDPipeline (ref2va) and MiniMaxH3ImageCodecOPSDPipeline (image)
    "ref2va": {"opcd_student_attention": "full", "condition_video_time": "sequential", "target_audio_rows": "silent"},
    "image": {"opcd_student_attention": "full", "target_audio_rows": "none"},
}
GAZERS = {"codec_gazer.fully_droppable_res_meanmax": "codec"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = ckpt["config"]
    model, adapter = config["model"], config["adapter"]
    variant = model.get("model_variant")
    if variant not in EXPECTED_MODEL:
        raise SystemExit(f"model_variant {variant!r} is not implemented here: one of {sorted(EXPECTED_MODEL)}")
    wrong = {k: model.get(k) for k, v in EXPECTED_MODEL[variant].items() if model.get(k) != v}
    # the student's VLM drop: picture cutoffs times this scale. MiniMax-H3 implements the scale of
    # configs/codec_gazer.yaml; MiniMax-H3-Image reads it from the LoRA's metadata.
    vlm_drop = bool(model.get("opcd_student_vlm_drop"))
    scale = model.get("opcd_student_vlm_picture_threshold_scale", 1.0)
    if vlm_drop and variant == "ref2va" and scale != load("codec_gazer")["vlm_drop"]["picture_threshold_scale"]:
        wrong["opcd_student_vlm_picture_threshold_scale"] = scale
    if adapter.get("vlm_vision_rows"):  # a LoRA on the text encoder's vision rows too: not implemented here
        wrong["adapter.vlm_vision_rows"] = True
    if adapter.get("free_row_lora_rank") or adapter.get("training_adapter") or adapter.get("lora_targets"):
        wrong["adapter"] = {k: adapter.get(k) for k in ("free_row_lora_rank", "training_adapter", "lora_targets")}
    if wrong:
        raise SystemExit(f"Checkpoint settings differ from this code: {wrong}")
    data = [task["data"] for task in config["tasks"]["definitions"].values()]
    gazers = {d.get("use_sparse_token") for d in data}
    # trained with other codec settings than codecopsd.codec_gazer.SETTINGS (threshold overrides, a token grid): refuse
    plain = {"version": "1.1.3", "video": None, "image": None}
    other = [d.get("codec_gazer_thresholds") for d in data if d.get("codec_gazer_thresholds") != plain]
    other += [d["token_grid"] for d in data if d.get("token_grid")]
    if len(gazers) != 1 or not gazers <= GAZERS.keys() or other:
        raise SystemExit(f"Token gazer not implemented by codecopsd.codec_gazer: {gazers}, {other}")

    tensors = {}
    for key, value in ckpt["lora_state_dict"].items():
        match = KEY.fullmatch(key)
        if match is None or match[2] not in MODULES:
            raise SystemExit(f"Unexpected tensor: {key}")
        part = "A" if match[3] == "down" else "B"
        tensors[f"transformer_blocks.{match[1]}.{match[2]}.lora_{part}.weight"] = value.contiguous()
    if len(tensors) != (49 * len(MODULES) + 2) * 2:  # the last block has only to_k / to_v
        raise SystemExit(f"Expected {(49 * len(MODULES) + 2) * 2} LoRA tensors, got {len(tensors)}")
    metadata = {
        "lora_rank": str(adapter["lora_rank"]), "lora_alpha": str(adapter["lora_alpha"]), "gazer": GAZERS[gazers.pop()],
        "vlm_drop": str(vlm_drop).lower(), "model": variant,
    }
    if vlm_drop:
        metadata["vlm_picture_threshold_scale"] = str(scale)
    save_file(tensors, args.output, metadata=metadata)
    print(f"{args.output}: {len(tensors)} tensors, " + ", ".join(f"{k} {v}" for k, v in metadata.items()))


if __name__ == "__main__":
    main()
