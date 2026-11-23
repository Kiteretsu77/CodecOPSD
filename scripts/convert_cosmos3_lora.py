"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Convert a Cosmos 3 CodecOPSD training checkpoint (.pt) into the LoRA file loaded by `load_condition_lora` (safetensors).

The training checkpoint also stores the full training config; only the LoRA rank / alpha, the token gazer and the
sampler settings are kept. The settings the pipeline implements are checked first, so a checkpoint trained differently
fails here instead of running with the wrong inputs. All-zero factors (the projections of the last layer whose condition
rows are never read) are left out.

    python scripts/convert_cosmos3_lora.py checkpoint.pt codecopsd_cosmos3.safetensors
"""

import argparse
import json
import re

import torch
from safetensors.torch import save_file

from codecopsd.configs import load

KEY = re.compile(r"layers\.(\d+)\.(self_attn\.(?:add_q_proj|add_k_proj|add_v_proj|to_add_out)|"
                 r"mlp_moe_gen\.(?:gate_proj|up_proj|down_proj))\.lora_([AB])\.weight")
GAZERS = {"codec_gazer.fully_droppable_res_meanmax": "codec"}
# Titan condition names of the training config -> the Cosmos 3 task / control they train (configs/cosmos3.yaml)
CONDITIONS = {"text": None, "cosmos_edge": "edge", "degradation": "blur", "cosmos_depth": "depth", "segmentation": "seg",
              "cosmos_i2v": "i2v", "cosmos_continuation": "continuation", "cosmos_v2a": "v2a",
              "cosmos_id_droid": "id_droid"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = ckpt["config"]
    model, adapter = config["model"], config["adapter"]
    cfg = load("cosmos3")
    expected = {"base_model": "cosmos3", "model_variant": "nano", "prompt_format": "framework",
                "flow_shift": cfg["flow_shift"], "generation_flow_shift": cfg["flow_shift"],
                "sampler_num_steps": cfg["num_inference_steps"]}
    wrong = {k: model.get(k) for k, v in expected.items() if model.get(k) != v}
    if adapter.get("type") != "cosmos3_lora":
        wrong["adapter.type"] = adapter.get("type")
    data = [task["data"] for task in config["tasks"]["definitions"].values()]
    gazers = {d.get("use_sparse_token") for d in data}
    plain = {"version": "1.1.3", "video": None, "image": None}
    other = [d.get("codec_gazer_thresholds") for d in data if d.get("codec_gazer_thresholds") != plain]
    other += [d["token_grid"] for d in data if d.get("token_grid")]
    if len(gazers) != 1 or not gazers <= GAZERS.keys() or other:
        wrong["gazer"] = (gazers, other)
    names = [c["name"] for d in data for c in d["conditions"]]
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        wrong["conditions"] = unknown
    if wrong:
        raise SystemExit(f"Checkpoint settings differ from this code: {wrong}")

    tensors, zero = {}, set()
    for key, value in ckpt["lora_state_dict"].items():
        match = KEY.fullmatch(key)
        if match is None:
            raise SystemExit(f"Unexpected tensor: {key}")
        if match[3] == "B" and not value.any():
            zero.add(key.rsplit(".lora_", 1)[0])
        tensors[key] = value.float().contiguous()
    for name in zero:
        del tensors[f"{name}.lora_A.weight"], tensors[f"{name}.lora_B.weight"]
    tasks = sorted({CONDITIONS[n] for n in names if CONDITIONS[n]})
    metadata = {
        "lora_rank": str(adapter["lora_rank"]), "lora_alpha": str(float(adapter["lora_alpha"])),
        "gazer": GAZERS[gazers.pop()], "tasks": json.dumps(tasks), "step": str(ckpt.get("step")),
    }
    save_file(tensors, args.output, metadata=metadata)
    print(f"{args.output}: {len(tensors)} tensors ({len(zero)} all-zero projections left out), "
          + ", ".join(f"{k} {v}" for k, v in metadata.items()))


if __name__ == "__main__":
    main()
