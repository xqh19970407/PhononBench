#!/usr/bin/env python
"""
Automatically generate Phonopy input files (.yaml.bz2) in batch for phonon calculations.

Usage example:
    python batch_prepare_phonopy_input.py --input_dir /path/to/cifs \
                                          --dim 2 2 2 \
                                          --out /path/to/output/

Supported formats:
    - CIF (.cif)
    - POSCAR / CONTCAR (.vasp)
    - XDATCAR / any structure file supported by ASE
"""


import os
import argparse
import phonopy
import bz2
from pathlib import Path
from ase.io import read
from phonopy.structure.atoms import PhonopyAtoms


def ase_to_phonopy(atoms):
    """ASE to PhonopyAtoms"""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions()
    )


def generate_phonopy_yaml(input_file, dim, out_dir):
    """GEN phonopy.yaml.bz2 """
    atoms = read(input_file)
    ph_atoms = ase_to_phonopy(atoms)

    ph = phonopy.Phonopy(
        unitcell=ph_atoms,
        supercell_matrix=dim
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    yaml_path = os.path.join(out_dir, f"{Path(input_file).stem}.yaml")
    bz2_path = yaml_path + ".bz2"

    # phonopy.yaml 
    ph.save(filename=yaml_path, settings={'force_constants': False})

    with open(yaml_path, "rb") as f_in:
        with bz2.open(bz2_path, "wb") as f_out:
            f_out.writelines(f_in)

    os.remove(yaml_path)
    print(f"✅ Generated: {bz2_path}")
    return bz2_path


def main():
    parser = argparse.ArgumentParser(description="Batch generate phonopy input YAML (.bz2) from a folder of structures")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing structure files (CIF, POSCAR, etc.)")
    parser.add_argument("--dim", type=int, nargs=3, default=[2, 2, 2], help="Supercell size, default = 2x2x2")
    parser.add_argument("--out", type=str, default="example/", help="Output folder (default: example/)")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out
    dim = args.dim

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith(('.cif','.vasp','.CONTCAR','.POSCAR'))]

    if not files:
        print(f"No structure files found in {input_dir}")
        return

    for f in files:
        input_file = os.path.join(input_dir, f)
        try:
            generate_phonopy_yaml(input_file, dim, out_dir)
        except Exception as e:
            print(f"❌ Error processing {f}: {e}")


if __name__ == "__main__":
    # python batch_prepare_phonopy_input.py --input_dir /home/xqhan/InvDesFlow3.0/Benchmark/MatterGen-gen/dft_band_gap/1.5/gen-cifs  --dim 2 2 2  --out /home/xqhan/InvDesFlow3.0/Benchmark/MatterGen-gen/dft_band_gap/bg_1.5/phonon-calculation-input
    main()
