"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Save the FP8 text encoder (VLM) once from the bf16 release (run from the repository root):

    python -m inference.vlm_fp8                      # models/MiniMax-H3-Ref2VA -> models/MiniMax-H3-VLM-FP8

Not needed when the FP8 VLM can be downloaded (`codecopsd.download.ensure_vlm_fp8`, the default of every entry point):
this is the fallback for a machine that cannot download it but has the bf16 release. It loads the bf16 text encoder as
the pipelines do (the 51 decoder layers MiniMax-H3 reads, converted by `codecopsd.models.fp8_linear.text_encoder_to_fp8`
on the GPU when there is one) and writes it as it is (`save_fp8_text_encoder`, 26 GB). The entry points then read that
folder as file views (`load_fp8_text_encoder`): the same weights, with no 66 GB bf16 read and no conversion at each
start. The conversion needs the CPU memory (and commit headroom) for the 66 GB bf16 files once, so run it where there
is enough.
"""

import argparse


def main(argv=None):
    from codecopsd.download import SAVED_VLM_FP8, model_dir

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(model_dir("MiniMax-H3-Ref2VA")),
                   help="the release folder, with text_encoder/")
    p.add_argument("--out", default=str(model_dir(SAVED_VLM_FP8)), help="the folder to write")
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForImageTextToText

    from codecopsd.configs import load
    from codecopsd.models.fp8_linear import save_fp8_text_encoder, text_encoder_to_fp8

    text_encoder = AutoModelForImageTextToText.from_pretrained(args.model, subfolder="text_encoder",
                                                               dtype=torch.bfloat16)
    keep = load("minimax_h3_ref2va")["text_encoder_layer"] + 1  # as the pipelines crop it
    del text_encoder.model.language_model.layers[keep:]
    text_encoder.config.text_config.num_hidden_layers = keep
    text_encoder_to_fp8(text_encoder, "fp8")
    print(f"saved {save_fp8_text_encoder(text_encoder, args.out)}: {keep} decoder layers, FP8")


if __name__ == "__main__":
    main()
