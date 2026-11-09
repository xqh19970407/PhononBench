#!/usr/bin/env python3
"""Bounded CPU/multi-GPU batches with per-material deadlines and verified skips."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def result_fields(out):
    data = json.loads((out / "summary.json").read_text())
    return {"formula": data["formula"], "dimension": data["dimension"],
            "relax_type": data["relax_type"], "pressure_GPa": data["pressure_GPa"],
            "minimum_band_THz": data["band_minimum_THz"], "minimum_mesh_THz": data["mesh_minimum_THz"],
            "energy_eV_cell": data["relaxation"]["energy_eV"],
            "enthalpy_eV_cell": data["relaxation"]["enthalpy_eV"],
            "volume_A3": data["final_cell_volume_A3"],
            "maximum_force_eV_A": data["relaxation"]["max_force_eV_A"],
            "mesh_negative_fraction_below_minus_0p1_THz": data["mesh_negative_mode_fraction_below_minus_0p1_THz"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", help="Allowed logical CPU list, e.g. 8,9,10,11")
    parser.add_argument("--gpus", help="Physical GPU indices, e.g. 2,3; one worker per GPU")
    parser.add_argument("--pressures", nargs="+", type=float, default=[0.0])
    parser.add_argument("--dimension", choices=["3d", "2d"], default="3d")
    parser.add_argument("--relax-type", choices=["none", "nvt", "npt"], default="npt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    cpus = [int(x) for x in args.cpus.split(",")] if args.cpus else available[:max(1, len(available)//2)]
    if not set(cpus) <= set(available) or len(cpus) != len(set(cpus)):
        parser.error("CPU list must contain distinct allowed logical CPUs")
    if min(args.workers, args.threads) < 1 or args.workers * args.threads > len(cpus):
        parser.error("workers * threads must fit the selected CPU budget")
    gpus = args.gpus.split(",") if args.gpus else []
    if gpus and (args.workers > len(gpus) or len(set(gpus)) != len(gpus)):
        parser.error("Supply at least one distinct GPU per worker")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest = args.manifest.resolve(strict=True)
        plan = json.loads(manifest.read_text())
        base = manifest.parent
        jobs, common = plan["jobs"], plan.get("common_args", [])
    else:
        base = args.input_dir.resolve(strict=True)
        common = ["--dimension", args.dimension, "--relax-type", args.relax_type]
        jobs = []
        for cif in sorted(base.rglob("*.cif")):
            relative = str(cif.relative_to(base))
            stem = cif.stem + "_" + hashlib.sha256(relative.encode()).hexdigest()[:8]
            for pressure in args.pressures:
                jobs.append({"id": stem + "_P" + str(pressure).replace(".", "p"),
                             "cif": str(cif), "args": ["--pressure", str(pressure)]})
    ids = [j["id"] for j in jobs]
    if not jobs or len(set(ids)) != len(ids) or any(Path(i).name != i or i in (".", "..") for i in ids):
        parser.error("Supply nonempty jobs with unique, simple directory IDs")
    for job in jobs:
        job["cif"] = str((base / job["cif"]).resolve(strict=True))
    plan = {"common_args": common, "jobs": jobs, "workers": args.workers,
            "threads": args.threads, "cpus": cpus, "gpus": gpus}
    (args.out / "batch_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return 0
    resources = queue.Queue()
    for index in range(args.workers):
        resources.put((cpus[index*args.threads:(index+1)*args.threads], gpus[index] if gpus else None))
    code_hash = digest(HERE / "phononbench.py") + ":" + digest(HERE / "paths_2d.py")
    runtime_versions = {name: importlib.metadata.version(name) for name in ["mattersim", "torch", "phonopy", "numpy", "ase", "spglib"]}
    model_hashes = {}
    def run(job):
        selected_cpus, gpu = resources.get()
        try:
            out = args.out / job["id"]
            options = list(common) + list(job.get("args", []))
            # Hardware and deadlines are controlled here; per-job physics is in options.
            if any(x in options for x in ["--_worker", "--out", "--cif", "--cpu-cores", "--threads", "--device"]):
                raise ValueError("Use batch resource flags rather than reserved per-job options")
            model = HERE.parent / "models/mattersim-v1.0.0-5M.pth"
            if "--model-path" in options:
                model = Path(options[len(options)-1-options[::-1].index("--model-path")+1]).resolve()
            model_key = str(model)
            if model_key not in model_hashes:
                model_hashes[model_key] = digest(model)
            spec = {"cif_sha256": digest(job["cif"]), "options": options,
                    "code_sha256": code_hash, "model_sha256": model_hashes[model_key],
                    "runtime_versions": runtime_versions,
                    "device": "cuda" if gpu is not None else "cpu"}
            submission = out / "batch_submission.json"
            state = out / "status.json"
            if submission.exists() and state.exists():
                old = json.loads(submission.read_text())
                if json.loads(state.read_text())["state"] == "DONE" and old.get("request") == spec:
                    archive = out / "PhononBench-results.zip"
                    if archive.is_file() and digest(archive) == old.get("archive_sha256"):
                        return {"id": job["id"], "state": "SKIPPED_DONE", "exit_code": 0,
                                "out": str(out), **result_fields(out)}
            if state.exists() and not args.resume:
                return {"id": job["id"], "state": "EXISTS_NEEDS_REVIEW", "exit_code": 2, "out": str(out)}
            out.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(HERE / "phononbench.py"), "--cif", job["cif"],
                       "--out", str(out), *options, "--threads", str(args.threads),
                       "--cpu-cores", ",".join(map(str, selected_cpus)),
                       "--device", "cuda" if gpu is not None else "cpu"]
            if args.resume: command.append("--resume")
            env = dict(os.environ)
            if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = gpu
            with (out / "submission.log").open("a") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = {"request": spec, "exit_code": result.returncode,
                      "ended_at": datetime.now(timezone.utc).isoformat()}
            archive = out / "PhononBench-results.zip"
            if result.returncode == 0 and archive.exists(): record["archive_sha256"] = digest(archive)
            submission.write_text(json.dumps(record, indent=2)+"\n")
            state_name = json.loads(state.read_text())["state"] if state.exists() else "FAILED"
            row = {"id": job["id"], "state": state_name, "exit_code": result.returncode, "out": str(out)}
            summary = out / "summary.json"
            if summary.exists() and result.returncode == 0:
                row.update(result_fields(out))
            return row
        finally:
            resources.put((selected_cpus, gpu))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            try: row = future.result()
            except Exception as error: row = {"id": futures[future]["id"], "state": "FAILED", "exit_code": 1, "error": str(error)}
            results.append(row)
            print(json.dumps(row), flush=True)
            (args.out / "batch_results.json").write_text(json.dumps(results, indent=2)+"\n")
    columns = sorted({k for row in results for k in row})
    with (args.out / "batch_results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(results)
    return int(any(row["exit_code"] for row in results))


if __name__ == "__main__":
    sys.exit(main())
