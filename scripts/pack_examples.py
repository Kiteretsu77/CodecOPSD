"""Pack the example media and the demo's stored example results into the two archives the demo downloads
(`codecopsd.download.EXAMPLE_ARCHIVES`), written at the repo root (out of git; paths as in the repo, so they unzip at
the repo root). Upload each to assets/ of its LoRA repo on Hugging Face (`codecopsd.download.LORA_REPOS`, docs/releasing.md).

    python scripts/pack_examples.py

The video archive holds the media of the video examples and the stored results of the video page and of its 12-32 GB
page; the image archive those of the image page. Media and results are already compressed, so the archives store
them. Derived copies the demo makes again on first launch are left out.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codecopsd.download import EXAMPLE_ARCHIVES, EXAMPLES, ROOT, example_files  # noqa: E402

DERIVED = (".fast.mp4", ".light.mp4", ".preview.mp4", ".preview.jpg", ".part")


def main():
    archives = {}
    for name, archive in EXAMPLE_ARCHIVES.items():
        archives.setdefault(archive, []).append(name)
    for archive, names in archives.items():
        refs = example_files(names)
        files = [EXAMPLES / "refs" / n for n in refs]
        missing = [f for f in files if not f.is_file()]
        if missing:
            raise SystemExit(f"{archive}: {len(missing)} example media files missing in assets/examples/refs/ "
                             f"(first: {missing[0].name})")
        for name in names:
            cache = EXAMPLES / "cache" / name
            if not any(cache.glob("*/meta.json")):
                raise SystemExit(f"{name}: no stored example results in {cache.relative_to(ROOT)}/ (--build-cache of "
                                 "its app)")
            files += [f for f in sorted(cache.rglob("*")) if f.is_file() and not f.name.endswith(DERIVED)
                      and ".before." not in f.name and "previews" not in f.relative_to(cache).parts
                      and f.name != ".release"]
        path = ROOT / archive
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for f in files:
                z.write(f, f.relative_to(ROOT).as_posix())
        print(f"{archive}: {len(refs)} media files + {len(files) - len(refs)} result files, "
              f"{path.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
