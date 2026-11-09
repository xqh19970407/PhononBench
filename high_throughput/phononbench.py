#!/usr/bin/env python3
"""Portable PhononBench calculations: 3D, 2D, NVT, NPT, and finite pressure."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
HARD_TIMEOUT_SECONDS = 1200


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def event(out, name, **data):
    row = {"time": now(), "event": name, **data}
    with (out / "events.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def status(out, state, **data):
    write_json(out / "status.json", {"updated_at": now(), "state": state, **data})


def read_atoms(path):
    from ase.io import read
    try:
        atoms = read(path)
        for occupancy in atoms.info.get("occupancy", {}).values():
            if len(occupancy) != 1 or abs(sum(occupancy.values()) - 1) > 1e-8:
                raise ValueError("Disordered occupancy")
        return atoms
    except Exception:
        from pymatgen.core import Structure
        from ase import Atoms
        s = Structure.from_file(path)
        if not s.is_ordered:
            raise ValueError("Choose an explicitly ordered configuration before calculation")
        return Atoms(symbols=[site.specie.symbol for site in s], positions=s.cart_coords,
                     cell=s.lattice.matrix, pbc=True)


def phonon_object(atoms, args):
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    ph = phonopy.Phonopy(PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array, scaled_positions=atoms.get_scaled_positions()),
        supercell_matrix=args.dim, primitive_matrix=None,
        is_symmetry=args.use_symmetry, symprec=1e-5)
    ph.generate_displacements(distance=args.distance, is_diagonal=False, is_plusminus=(True if args.displacement_signs == "both" else "auto"))
    return ph


def preflight(args):
    import numpy as np
    import spglib
    a = read_atoms(args.cif)
    dataset = spglib.get_symmetry_dataset((a.cell.array, a.get_scaled_positions(), a.numbers), symprec=1e-5)
    distances = a.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    from ase.geometry import minkowski_reduce
    reduced, _ = minkowski_reduce(a.cell, pbc=True)
    minimum_distance = min(float(distances.min()), float(np.linalg.norm(reduced, axis=1).min()))
    ph = phonon_object(a, args)
    return {"input": str(args.cif), "input_sha256": sha256(args.cif),
            "formula": a.get_chemical_formula(), "unitcell_atoms": len(a),
            "cell_lengths_A": a.cell.lengths().tolist(), "cell_angles_deg": a.cell.angles().tolist(),
            "space_group": dataset.international, "space_group_number": int(dataset.number),
            "symmetry_operations": len(dataset.rotations), "use_symmetry": args.use_symmetry,
            "minimum_distance_A": minimum_distance, "supercell_atoms": len(ph.supercell),
            "supercell_matrix": args.dim, "displacement_count": len(ph.displacements),
            "displacement_signs": args.displacement_signs, "dimension": args.dimension, "distance_A": args.distance,
            "relaxation": args.relax, "hard_timeout_seconds": args.timeout_seconds}


class ThermalGate:
    def __init__(self, out):
        self.out, self.last = out, 0.0
        self.limit = None

    def __call__(self):
        if self.limit is None or time.monotonic() - self.last < 10:
            return
        import psutil
        self.last = time.monotonic()
        def temperature():
            values = [t.current for name, sensors in psutil.sensors_temperatures().items()
                      if name in ("coretemp", "k10temp", "zenpower") for t in sensors]
            if not values:
                return None
            return max(values)
        t = temperature()
        event(self.out, "resource_sample", cpu_max_C=t,
              rss_MiB=psutil.Process().memory_info().rss / 1024**2)
        if t is not None and t >= self.limit:
            event(self.out, "cooling_wait", temperature_C=t)
            while t is not None and t > min(60, self.limit - 5):
                time.sleep(5)
                t = temperature()
            event(self.out, "cooling_finished", temperature_C=t)


def worker(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import phonopy
    import psutil
    import torch
    from ase import Atoms
    from ase.io import write, read
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from ase.constraints import FixSymmetry
    from ase.units import GPa
    from mattersim.forcefield import MatterSimCalculator

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    out = args.out
    gate = ThermalGate(out)
    gate.limit = args.temperature_limit
    gate()
    model_path = args.model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}; see models/README.md")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing a silent CPU fallback")
    calculation = {"input_sha256": sha256(args.cif), "workflow_sha256": sha256(__file__),
        "path_generator_sha256": sha256(HERE / "paths_2d.py"),
        "runtime_versions": {"phonopy": phonopy.__version__, "numpy": np.__version__, "torch": torch.__version__},
        "model_sha256": sha256(model_path), "model": args.model, "model_path": str(model_path),
        "device": args.device, "dim": args.dim, "distance": args.distance,
        "relax": args.relax, "fmax": args.fmax, "steps": args.steps,
        "pressure_GPa": args.pressure, "mesh": args.mesh, "pdos": args.pdos,
        "use_symmetry": args.use_symmetry, "fix_symmetry": args.fix_symmetry,
        "dimension": args.dimension, "cell_policy": args.cell_policy,
        "displacement_signs": args.displacement_signs, "mesh_density": args.mesh_density,
        "dos_method": args.dos_method, "frequency_pitch_THz": args.frequency_pitch, "npoints": args.npoints}
    config_path = out / "calculation.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != calculation:
            raise RuntimeError("Checkpoint configuration/input/model/code differs; use a new output directory")
    else:
        write_json(config_path, calculation)
    calc = MatterSimCalculator(load_path=str(model_path), device=args.device)
    initial = read_atoms(args.cif)
    a = initial.copy()
    a.calc = calc
    if args.dimension == "2d" and args.cell_policy == "slab":
        if not np.allclose(a.cell.array[:2, 2], 0, atol=1e-7) or not np.allclose(a.cell.array[2, :2], 0, atol=1e-7):
            raise ValueError("Slab policy requires ab in xy and c perpendicular to ab")
    initial_forces = a.get_forces()
    initial_energy = float(a.get_potential_energy())
    input_info = preflight(args)
    input_info.update(initial_energy_eV=initial_energy,
        initial_max_force_eV_A=float(np.linalg.norm(initial_forces, axis=1).max()))
    write_json(out / "input_analysis.json", input_info)
    write(out / "input.cif", initial)
    event(out, "initial_force", **input_info)
    status(out, "RUNNING", stage="relaxation")
    relaxation_path = out / "relaxation.json"
    if args.resume and relaxation_path.exists() and (out / "relaxed_geometry.npz").exists():
        with np.load(out / "relaxed_geometry.npz", allow_pickle=False) as geometry:
            a = Atoms(numbers=geometry["numbers"], positions=geometry["positions"],
                      cell=geometry["cell"], pbc=geometry["pbc"])
        a.calc = calc
        if args.fix_symmetry and args.relax != "none":
            a.set_constraint(FixSymmetry(a, symprec=1e-5))
        relaxation = json.loads(relaxation_path.read_text())
        event(out, "relaxation_checkpoint_reused")
    else:
        steps, converged = 0, True
        if args.relax != "none":
            if args.fix_symmetry:
                a.set_constraint(FixSymmetry(a, symprec=1e-5))
            mask = [True, True, False, False, False, True] if args.dimension == "2d" and args.cell_policy == "slab" else [True] * 6
            target = a if args.relax == "positions" else FrechetCellFilter(a, mask=mask, scalar_pressure=args.pressure * GPa)
            opt = FIRE(target, logfile=str(out / "relax.log"), trajectory=str(out / "relax.traj"))
            def on_step():
                gate()
                if opt.nsteps % 25 == 0:
                    f = target.get_forces()
                    event(out, "relaxation_progress", step=opt.nsteps,
                          max_optimizer_force=float(np.linalg.norm(f, axis=1).max()))
            opt.attach(on_step, interval=1)
            converged = bool(opt.run(fmax=args.fmax / 2, steps=args.steps))
            steps = opt.nsteps
        forces = a.get_forces()
        relaxation = {"mode": args.relax, "converged": converged,
            "steps": steps, "energy_eV": float(a.get_potential_energy()),
            "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
            "stress_GPa": (a.get_stress() / GPa).tolist(),
            "raw_max_force_eV_A": float(np.linalg.norm(a.get_forces(apply_constraint=False), axis=1).max()),
            "pressure_GPa": args.pressure, "enthalpy_eV": float(a.get_potential_energy() + args.pressure * GPa * a.get_volume()),
            "target_optimizer_force_eV_A": args.fmax / 2, "cell_policy": args.cell_policy,
            "cell_lengths_A": a.cell.lengths().tolist(), "cell_angles_deg": a.cell.angles().tolist()}
        write(out / "relaxed.cif", a)
        # Geometry-only checkpoint: extxyz cannot serialize FixSymmetry's mask.
        # Constraints and the calculator are reconstructed explicitly on resume.
        checkpoint = Atoms(symbols=a.get_chemical_symbols(), positions=a.positions,
                           cell=a.cell, pbc=a.pbc)
        write(out / "relaxed.extxyz", checkpoint)
        np.savez(out / "relaxed_geometry.npz", numbers=a.numbers, positions=a.positions,
                 cell=a.cell.array, pbc=a.pbc)
        if not converged:
            raise RuntimeError("Relaxation did not converge; phonons were not calculated")
        write_json(relaxation_path, relaxation)
    if args.relax == "positions" and not np.allclose(a.cell.array, initial.cell.array, atol=1e-10):
        raise RuntimeError("Fixed-cell relaxation changed the lattice")
    if args.relax != "none" and np.linalg.norm(a.get_forces(), axis=1).max() > args.fmax * 1.001:
        raise RuntimeError("Relaxed structure exceeds force threshold")
    if args.dimension == "2d" and args.cell_policy == "slab" and not np.allclose(a.cell.array[2], initial.cell.array[2], atol=1e-8):
        raise RuntimeError("Slab relaxation changed the vacuum vector")
    event(out, "relaxation_done", **relaxation)
    ph = phonon_object(a, args)
    displacements = np.asarray(ph.displacements)
    np.save(out / "displacements.npy", displacements)
    forces_dir = out / "forces"
    forces_dir.mkdir(exist_ok=True)
    forcesets = []
    force_start = time.monotonic()
    for i, sc in enumerate(ph.supercells_with_displacements):
        gate()
        path = forces_dir / f"{i:04d}.npz"
        if args.resume and path.exists():
            with np.load(path) as saved:
                force = saved["forces"].copy()
            reused = True
        else:
            atoms = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
            atoms.calc = calc
            raw = atoms.get_forces()
            drift = raw.mean(axis=0)
            force = raw - drift
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.savez(f, forces=force, raw_forces=raw, drift=drift)
            os.replace(temporary, path)
            reused = False
        if force.shape != (len(sc), 3) or not np.isfinite(force).all():
            raise RuntimeError(f"Invalid forces at displacement {i}")
        forcesets.append(force)
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(displacements):
            event(out, "force_progress", completed=i+1, total=len(displacements), reused_last=reused,
                  seconds_since_force_loop=round(time.monotonic()-force_start, 2))
            status(out, "RUNNING", stage="forces", completed=i+1, total=len(displacements))
    ph.forces = forcesets
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    status(out, "RUNNING", stage="bands_and_dos")
    gate()
    # Phonopy 2.47's YAML dumper expects crystallographic attributes that its
    # NosymDataset does not provide. Omit that optional metadata section while
    # retaining the actual cells, displacements, forces and force constants.
    from phonopy.interface.phonopy_yaml import PhonopyYaml
    yaml = PhonopyYaml(settings={"force_constants": True})
    yaml.set_phonon_info(ph)
    if not args.use_symmetry:
        yaml.symmetry = None
    (out / "phonopy_fc.yaml").write_text(str(yaml))
    if args.dimension == "2d":
        from paths_2d import band_path_2d
        paths, labels, connections = band_path_2d(ph.primitive.cell, args.npoints)
        ph.run_band_structure(paths, labels=labels, path_connections=connections)
        ph.write_yaml_band_structure(filename=str(out / "phononbench_band.yaml"))
    else:
        ph.auto_band_structure(npoints=args.npoints, with_eigenvectors=False, plot=False,
                              write_yaml=True, filename=str(out / "phononbench_band.yaml"))
    band = ph.get_band_structure_dict()
    distances = np.concatenate(band["distances"])
    frequencies = np.vstack(band["frequencies"])
    qpoints = np.vstack(band["qpoints"])
    np.savetxt(out / "phonon_band.dat", np.column_stack([distances, frequencies]),
               header="distance followed by sorted phonon frequencies (THz)")
    np.savetxt(out / "band_qpoints.dat", np.column_stack([distances, qpoints]),
               header="distance qx qy qz (fractional reciprocal coordinates)")
    picture = ph.plot_band_structure()
    picture.savefig(out / "phonon_band.png", dpi=220)
    picture.savefig(out / "phonon_band.pdf")
    plt.close("all")
    gate()
    from phonopy.structure.grid_points import length2mesh
    mesh_numbers = list(args.mesh) if args.mesh else list(map(int, length2mesh(args.mesh_density, ph.primitive.cell)))
    if args.dimension == "2d":
        mesh_numbers[2] = 1
    ph.run_mesh(mesh_numbers, with_eigenvectors=args.pdos, is_mesh_symmetry=False,
                is_time_reversal=False, is_gamma_center=False)
    mesh = ph.get_mesh_dict()
    np.savez_compressed(out / "mesh_frequencies.npz", qpoints=mesh["qpoints"],
                        frequencies=mesh["frequencies"], weights=mesh["weights"])
    dos_method = args.dos_method
    if dos_method == "auto":
        dos_method = "gaussian" if args.dimension == "2d" else "tetrahedron"
    tetrahedron = dos_method == "tetrahedron"
    ph.run_total_dos(sigma=None if tetrahedron else 0.05,
                     freq_pitch=args.frequency_pitch, use_tetrahedron_method=tetrahedron)
    ph.write_total_dos(filename=out / "total_dos.dat")
    picture = ph.plot_total_dos()
    picture.savefig(out / "phonon_dos.png", dpi=220)
    plt.close("all")
    if args.pdos:
        ph.run_projected_dos(sigma=None if tetrahedron else 0.05,
                             freq_pitch=args.frequency_pitch, use_tetrahedron_method=tetrahedron)
        ph.write_projected_dos(filename=out / "projected_dos.dat")
        elements = list(dict.fromkeys(a.get_chemical_symbols()))
        indices = [[i for i, s in enumerate(a.get_chemical_symbols()) if s == e] for e in elements]
        picture = ph.plot_projected_dos(pdos_indices=indices, legend=elements)
        picture.savefig(out / "projected_dos.png", dpi=220)
        plt.close("all")
    mf = mesh["frequencies"]
    q, mode = np.unravel_index(np.argmin(mf), mf.shape)
    summary = {"completed_at": now(), "method": "MatterSim finite displacements + Phonopy; not DFT",
        "formula": a.get_chemical_formula(), "natoms": len(a), "supercell_atoms": len(ph.supercell),
        "displacements": len(displacements), "distance_A": args.distance,
        "initial_cell_volume_A3": float(initial.get_volume()), "final_cell_volume_A3": float(a.get_volume()),
        "initial_max_force_eV_A": input_info["initial_max_force_eV_A"],
        "relaxation": relaxation, "energy_per_atom_eV": relaxation["energy_eV"] / len(a),
        "band_minimum_THz": float(frequencies.min()), "mesh_minimum_THz": float(mf[q, mode]),
        "mesh_minimum_q": mesh["qpoints"][q].tolist(), "mesh_minimum_mode_1based": int(mode+1),
        "mesh": mesh_numbers, "mesh_qpoints": len(mf),
        "dimension": args.dimension, "relax_type": args.relax_type, "fix_symmetry": args.fix_symmetry,
        "cell_policy": args.cell_policy, "pressure_GPa": args.pressure, "DOS_method": dos_method,
        "frequency_pitch_THz": args.frequency_pitch,
        "qz_maximum_absolute": float(np.abs(mesh["qpoints"][:, 2]).max()),
        "mesh_negative_mode_fraction_below_minus_0p1_THz": float((mf < -0.1).mean()),
        "mesh_negative_mode_fraction_below_minus_0p5_THz": float((mf < -0.5).mean()),
        "mesh_qpoint_fraction_with_mode_below_minus_0p1_THz": float((mf.min(axis=1) < -0.1).mean()),
        "Gamma_frequencies_THz": ph.get_frequencies([0, 0, 0]).tolist(),
        "phonopy_version": phonopy.__version__, "torch_version": torch.__version__,
        "cpu_affinity": psutil.Process().cpu_affinity(), "threads": args.threads,
        "device": args.device, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_model": str(model_path), "checkpoint_sha256": calculation["model_sha256"]}
    write_json(out / "summary.json", summary)
    (out / "summary.txt").write_text(
        f"Formula: {summary['formula']}\nDimension: {args.dimension}\nRelaxation: {args.relax_type}\nPressure: {args.pressure} GPa\n"
        f"Model: {args.model} ({args.device}); not DFT\n"
        f"Maximum final force: {relaxation['max_force_eV_A']:.6f} eV/Angstrom\n"
        f"Minimum band frequency: {summary['band_minimum_THz']:.6f} THz\n"
        f"Minimum mesh frequency: {summary['mesh_minimum_THz']:.6f} THz\n"
        "Negative frequencies denote imaginary modes. Small near-zero acoustic negatives require convergence checks.\n"
        "This finite-supercell harmonic result does not establish finite-temperature or thermodynamic stability.\n")
    bundle_files = [p for p in out.iterdir() if p.is_file() and
        p.name not in {"status.json", "events.jsonl", "relax.traj", "relaxed.extxyz", "worker.log"} and
        p.suffix in {".cif", ".json", ".txt", ".yaml", ".npy", ".npz", ".dat", ".png", ".pdf"}]
    with zipfile.ZipFile(out / "PhononBench-results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in bundle_files:
            z.write(p, p.name)
    event(out, "calculation_completed", band_min_THz=summary["band_minimum_THz"],
          mesh_min_THz=summary["mesh_minimum_THz"])


def terminate_group(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def arm_worker_guard(seconds):
    """Keep the deadline valid even if the supervising shell/process disappears."""
    import ctypes
    expected = int(os.environ.get("PHONONBENCH_SUPERVISOR_PID", "0"))
    if expected <= 1 or os.getppid() != expected:
        raise RuntimeError("Worker requires its live supervisor")
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux PR_SET_PDEATHSIG: terminate immediately if the supervisor dies.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot arm parent-death guard")
    if os.getppid() != expected:
        os.kill(os.getpid(), signal.SIGKILL)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(seconds)


def supervise(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    state = out / "status.json"
    if state.exists() and not args.resume:
        raise RuntimeError("Output already contains a job; use a new output or explicitly --resume")
    status(out, "QUEUED", input=str(args.cif), timeout_seconds=args.timeout_seconds)
    event(out, "queued", input=str(args.cif), timeout_seconds=args.timeout_seconds)
    with (out / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
        child = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            with (out / "worker.log").open("a") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PHONONBENCH_SUPERVISOR_PID=str(os.getpid())))
                status(out, "RUNNING", worker_pid=child.pid, timeout_seconds=args.timeout_seconds)
                event(out, "worker_started", worker_pid=child.pid)
                try:
                    code = child.wait(timeout=args.timeout_seconds)
                    if code == -signal.SIGALRM:
                        status(out, "TIMEOUT", elapsed_seconds=round(time.monotonic()-started, 2),
                               timeout_seconds=args.timeout_seconds)
                        event(out, "timeout", timeout_seconds=args.timeout_seconds)
                        return 124
                except subprocess.TimeoutExpired:
                    terminate_group(child)
                    status(out, "TIMEOUT", elapsed_seconds=round(time.monotonic()-started, 2),
                           timeout_seconds=args.timeout_seconds)
                    event(out, "timeout", timeout_seconds=args.timeout_seconds)
                    return 124
            elapsed = round(time.monotonic()-started, 2)
            status(out, "DONE" if code == 0 else "FAILED", exit_code=code, elapsed_seconds=elapsed)
            event(out, "finished", exit_code=code, elapsed_seconds=elapsed)
            return code
        except KeyboardInterrupt:
            if child is not None:
                terminate_group(child)
            status(out, "CANCELLED", elapsed_seconds=round(time.monotonic()-started, 2))
            return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dim", type=int, nargs=3, default=None)
    parser.add_argument("--model", default="mattersim-v1")
    parser.add_argument("--model-path", type=Path, default=REPOSITORY / "models/mattersim-v1.0.0-5M.pth")
    parser.add_argument("--dimension", choices=["3d", "2d"], default="3d")
    parser.add_argument("--cell-policy", choices=["slab", "web"], default="slab")
    parser.add_argument("--fix-symmetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--displacement-signs", choices=["auto", "both"], default="auto")
    parser.add_argument("--mesh-density", type=float, default=100.0)
    parser.add_argument("--dos-method", choices=["auto", "tetrahedron", "gaussian"], default="auto")
    parser.add_argument("--frequency-pitch", type=float, default=0.01)
    parser.add_argument("--temperature-limit", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-cores", help="Logical CPU list, for example 8,9")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--relax-type", choices=["none", "nvt", "npt"], default="npt")
    parser.add_argument("--fmax", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=7500)
    parser.add_argument("--pressure", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--mesh", type=int, nargs=3)
    parser.add_argument("--npoints", type=int, default=101)
    parser.add_argument("--pdos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-symmetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=int, default=HARD_TIMEOUT_SECONDS)
    parser.add_argument("--preflight", action="store_true", help="Inspect CIF/displacements without force calculation")
    parser.add_argument("--resume", action="store_true", help="Explicitly reuse matching completed checkpoints")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= HARD_TIMEOUT_SECONDS:
        parser.error("Timeout must be between 1 and 1200 seconds")
    args.relax = {"none": "none", "nvt": "positions", "npt": "cell"}[args.relax_type]
    if args.dim is None:
        args.dim = [4, 4, 1] if args.dimension == "2d" else [2, 2, 2]
    if args.dimension == "2d" and args.dim[2] != 1:
        parser.error("2D requires a supercell with dim[2]=1")
    if args.pressure and args.relax_type != "npt":
        parser.error("Nonzero pressure requires --relax-type npt")
    if args.dimension == "2d" and args.pressure and args.cell_policy != "web":
        parser.error("3D pressure in a vacuum slab requires explicit --cell-policy web; see documentation")
    args.model_path = args.model_path.resolve()
    if args.threads < 1 or min(args.dim + (args.mesh or [1, 1, 1])) < 1 or args.distance <= 0 or args.fmax <= 0:
        parser.error("Threads, dimensions, mesh, distance and force threshold must be positive")
    if args.steps < 1 or args.npoints < 2 or args.mesh_density <= 0 or args.frequency_pitch <= 0:
        parser.error("Steps and mesh density must be positive; npoints must be at least 2")
    if not all(math.isfinite(x) for x in [args.pressure, args.distance, args.fmax, args.mesh_density, args.frequency_pitch]):
        parser.error("Numeric calculation parameters must be finite")
    args.cif, args.out = args.cif.resolve(strict=True), args.out.resolve()
    available = sorted(os.sched_getaffinity(0))
    selected = {int(x) for x in args.cpu_cores.split(",")} if args.cpu_cores else set(available[:args.threads])
    if not selected <= set(available):
        parser.error("Requested CPU cores are outside the allowed affinity")
    if args.threads > len(selected):
        parser.error("Threads cannot exceed the selected logical CPUs")
    os.sched_setaffinity(0, selected)
    os.environ.update(TZ="Asia/Shanghai", PYTHONNOUSERSITE="1", MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    time.tzset()
    if args.preflight:
        print(json.dumps(preflight(args), ensure_ascii=False, indent=2))
        return 0
    if args._worker:
        arm_worker_guard(args.timeout_seconds)
        worker(args)
        signal.alarm(0)
        return 0
    return supervise(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
