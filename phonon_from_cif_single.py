#!/usr/bin/env python
"""
phonon_from_cif_single.py
-------------------------
从单个 CIF 文件出发：
1) 生成 phonopy.yaml.bz2
2) MLIP 弛豫
3) 声子计算
4) 画声子谱
5) 打包结果为 zip

输出 zip 内包含：
- *_phonon_band.png
- *_phonopy_fc.yaml # 太大,不打包
- *_band.npy
- *_relaxed.cif
- summary.txt
"""

import os
import sys
import bz2
import shutil
import zipfile
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import phonopy
from ase import Atoms
from ase.io import read
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
from umlip import umlip


# ============================================================
# CIF → phonopy.yaml.bz2（原封不动）
# ============================================================
def ase_to_phonopy(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions()
    )


def generate_phonopy_yaml(cif_file, dim, out_dir):
    atoms = read(cif_file)
    ph_atoms = ase_to_phonopy(atoms)

    ph = phonopy.Phonopy(
        unitcell=ph_atoms,
        supercell_matrix=dim
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = Path(cif_file).stem
    yaml_path = out_dir / f"{name}.yaml"
    bz2_path = out_dir / f"{name}.yaml.bz2"

    ph.save(filename=str(yaml_path), settings={"force_constants": False})

    with open(yaml_path, "rb") as f_in:
        with bz2.open(bz2_path, "wb") as f_out:
            f_out.writelines(f_in)

    yaml_path.unlink()
    return bz2_path


# ============================================================
# phonopy + MLIP 计算（完全不改）
# ============================================================
def run_phonopy(ph_ref, mlip, distance=0.01, relax=True):
    ase_cell = Atoms(
        cell=ph_ref.unitcell.cell,
        symbols=ph_ref.unitcell.symbols,
        scaled_positions=ph_ref.unitcell.scaled_positions,
        pbc=True
    )

    if relax:
        relaxed = mlip.relax_structure(
            ase_cell,
            fmax=0.005,
            check_cell=False,
            check_connected=False,
            fix_symmetry=True
        )
        if relaxed is None:
            return None
        ase_cell = relaxed[0]
    else:
        relaxed = [ase_cell, None, None]

    ph_atoms = PhonopyAtoms(
        cell=ase_cell.get_cell(),
        scaled_positions=ase_cell.get_scaled_positions(),
        symbols=ase_cell.get_chemical_symbols()
    )

    ph = phonopy.Phonopy(
        ph_atoms,
        supercell_matrix=ph_ref.supercell_matrix,
        primitive_matrix=ph_ref.primitive_matrix
    )

    ph.generate_displacements(distance=distance, is_diagonal=False)

    forcesets = []
    for supercell in ph.supercells_with_displacements:
        scell = Atoms(
            cell=supercell.cell,
            symbols=supercell.symbols,
            scaled_positions=supercell.scaled_positions,
            pbc=True
        )
        scell.calc = mlip.calculator
        forces = scell.get_forces()
        forces -= forces.sum(axis=0) / forces.shape[0]
        forcesets.append(forces)

    ph.forces = forcesets
    ph.produce_force_constants()
    ph.symmetrize_force_constants()

    return ph, relaxed


# ============================================================
# 声子谱计算 + 画图（一行未动）
# ============================================================
def calculate_and_plot_band_structure(ph, structure_name, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bands, labels, path_connections = get_band_qpoints_by_seekpath(
        ph.primitive,
        npoints=101,
        is_const_interval=True
    )

    ph.run_band_structure(
        bands,
        path_connections=path_connections,
        labels=labels,
        is_legacy_plot=False
    )

    band_dict = ph.get_band_structure_dict()

    all_distances = []
    all_frequencies = []
    special_points = []
    used_labels = []

    for i, connected in enumerate(path_connections):
        if connected:
            all_distances.append(band_dict['distances'][i])
            all_frequencies.append(band_dict['frequencies'][i])
            special_points.extend([
                band_dict['distances'][i][0],
                band_dict['distances'][i][-1]
            ])
            used_labels.extend([labels[i], labels[i + 1]])

    all_distances = np.concatenate(all_distances)
    all_frequencies = np.vstack(all_frequencies)

    plt.figure(figsize=(10, 8))
    for i in range(all_frequencies.shape[1]):
        plt.plot(all_distances, all_frequencies[:, i], 'b-', linewidth=1)

    seen = set()
    sp_unique, lb_unique = [], []
    for sp, lb in zip(special_points, used_labels):
        if sp not in seen:
            sp_unique.append(sp)
            lb_unique.append(lb)
            seen.add(sp)

    for sp in sp_unique:
        plt.axvline(sp, color='k', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.xticks(sp_unique, lb_unique, fontsize=14)
    plt.ylabel('Frequency (THz)')
    plt.xlabel('Wave Vector')
    plt.title(structure_name)
    plt.xlim(all_distances.min(), all_distances.max())
    plt.tight_layout()

    plot_path = Path(output_dir) / f"{structure_name}_phonon_band.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return band_dict


# ============================================================
# 主流程 + zip 打包
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True)
    parser.add_argument("--dim", type=int, nargs=3, default=[2, 2, 2])
    parser.add_argument("--model", default="mattersim-v1")
    parser.add_argument("--out", default="phonon_output")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = Path(args.cif).stem

    yaml_bz2 = generate_phonopy_yaml(args.cif, args.dim, out_dir)
    ph_ref = phonopy.load(yaml_bz2)

    mlip = umlip(model=args.model)
    result = run_phonopy(ph_ref, mlip)

    if result is None:
        sys.exit("❌ Relaxation failed")

    ph, relaxed = result

    relaxed_cif = out_dir / f"{name}_relaxed.cif"
    relaxed[0].write(relaxed_cif)

    fc_yaml = out_dir / f"{name}_phonopy_fc.yaml"
    ph.save(filename=str(fc_yaml), settings={"force_constants": True})

    band_data = calculate_and_plot_band_structure(ph, name, out_dir)
    np.save(out_dir / f"{name}_band.npy", band_data)

    all_freqs = np.concatenate(band_data["frequencies"])
    stable = not np.any(all_freqs < -1e-3)

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Structure: {name}\n")
        # f.write(f"Model: {args.model}\n")
        f.write(f"Status: {'Stable' if stable else 'Unstable'}\n")

    # ======================
    # 打包 zip（白名单）
    # ======================
    zip_path = out_dir / f"{name}_phonon_results.zip"

    files_to_zip = [
        out_dir / f"{name}_relaxed.cif",
        out_dir / f"{name}_phonon_band.png",
        out_dir / f"{name}_band.npy",
        out_dir / "summary.txt",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file in files_to_zip:
            if file.exists():
                z.write(file, arcname=file.name)

    print(f"🎁 Results packaged: {zip_path}")



if __name__ == "__main__":
    # python phonon_from_cif_single.py --cif /src/Si.cif --dim 2 2 2 --model mattersim-v1 --out /test0106
    main()
