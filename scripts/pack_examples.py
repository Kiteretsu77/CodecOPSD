"""Pack the example media and the demo's stored example results into the zip archives of the GitHub Release
`codecopsd.download.EXAMPLES_TAG`, written to release_assets/ (out of git). Upload the five files to that release; every
install of the demo then downloads them once on its first launch (`codecopsd.download.ensure_examples`).

    python scripts/pack_examples.py

Per model: codecopsd_refs_<model>_<tag>.zip (refs/<file>, the media its examples refer to) and
codecopsd_cache_<model>_<tag>.zip (cache/<model>/..., the results the demo shows when an example is clicked, plus a
cache/<model>/.release file naming the tag). A model that reuses another one's examples (`EXAMPLES_OF`: the extreme
page) has no media archive of its own. Media and results are already compressed, so the archives store them.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codecopsd.download import EXAMPLES, EXAMPLES_OF, EXAMPLES_TAG, NEEDS, ROOT, archive_name, example_files  # noqa: E402


def main():
    out = ROOT / "release_assets"
    out.mkdir(exist_ok=True)
    for name in NEEDS:
        refs = [] if name in EXAMPLES_OF else example_files([name])
        missing = [n for n in refs if not (EXAMPLES / "refs" / n).is_file()]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} example media files missing in assets/examples/refs/ "
                             f"(first: {missing[0]})")
        if refs:
            path = out / archive_name("refs", name)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
                for n in refs:
                    z.write(EXAMPLES / "refs" / n, f"refs/{n}")
            print(f"{path.relative_to(ROOT)}: {len(refs)} files, {path.stat().st_size / 2**20:.0f} MB")

        cache = EXAMPLES / "cache" / name
        stored = sorted(cache.glob("*/meta.json"))
        if not stored:
            raise SystemExit(f"{name}: no stored example results in {cache.relative_to(ROOT)}/ (--build-cache of its "
                             "app)")
        path = out / archive_name("cache", name)
        files = [f for f in sorted(cache.rglob("*")) if f.is_file() and f.name != ".release" and f.suffix != ".part"]
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for f in files:
                z.write(f, f"cache/{name}/{f.relative_to(cache).as_posix()}")
            z.writestr(f"cache/{name}/.release", EXAMPLES_TAG + "\n")
        print(f"{path.relative_to(ROOT)}: {len(stored)} examples, {len(files)} files, "
              f"{path.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
