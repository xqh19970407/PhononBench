#!/usr/bin/env python3
"""Build a materialized code/model ZIP from tracked files, excluding runtime data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = ROOT / "models/mattersim-v1.0.0-5M.pth"
    expected = json.loads((ROOT / "models/manifest.json").read_text())
    if digest(model) != expected["sha256"]:
        raise RuntimeError("Model does not match manifest; run git lfs pull if it is a pointer")
    paths = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")
    files = [p for p in paths if p and not p.startswith(("distributions/", "results/"))]
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "source_code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "files": {p: {"bytes": (ROOT/p).stat().st_size, "sha256": digest(ROOT/p)} for p in files}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files: archive.write(ROOT/p, "PhononBench/"+p)
        archive.writestr("PhononBench/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(args.out) as archive:
        assert archive.testzip() is None
        assert archive.getinfo("PhononBench/models/"+model.name).file_size == model.stat().st_size
    checksum = digest(args.out)
    args.out.with_suffix(".zip.sha256").write_text(checksum+"  "+args.out.name+"\n")
    print(json.dumps({"zip": str(args.out.resolve()), "bytes": args.out.stat().st_size,
                      "sha256": checksum, "files": len(files)+1}, indent=2))


if __name__ == "__main__": main()
