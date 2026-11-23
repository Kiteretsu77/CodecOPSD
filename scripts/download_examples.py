"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Download the reference pictures and videos of the bundled examples into assets/examples/refs/ (the demos, app.py, and
inference/generate.py also do it on first use; `python -m codecopsd.download` fetches the models too).

    python scripts/download_examples.py

The examples themselves (assets/examples/<model>/<case>/example.json + prompt.txt) are in the repository; their media
(assets/examples/references.txt, about 200 MB; the demo examples alone about 120 MB) are not. Files already present are
skipped.
"""

from codecopsd.download import ensure_examples

if __name__ == "__main__":
    print(f"example media in {ensure_examples()}")
