import sys
import os
import glob
import random
import yaml
import numpy as np
from ase import Atoms
from pathlib import Path

import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
import matplotlib.pyplot as plt

from umlip import umlip  # Your MLIP module

def run_phonopy(ph_ref, mlip, distance=0.01, relax=True):
    """Run phonopy and compute the phonon spectrum"""
    # Create ASE structure
    ase_cell = Atoms(
        cell=ph_ref.unitcell.cell,
        symbols=ph_ref.unitcell.symbols,
        scaled_positions=ph_ref.unitcell.scaled_positions,
        pbc=True)

    # Relax the unit cell
    if relax:
        relaxed = mlip.relax_structure(ase_cell, fmax=0.005, check_cell=False,
                                     check_connected=False, fix_symmetry=True)
        if relaxed is None:
            return None
        ase_cell = relaxed[0]
    else:
        relaxed = [ase_cell, None, None]

    # Create phonopy structure
    ph_atoms = phonopy.structure.atoms.PhonopyAtoms(
        cell=ase_cell.get_cell(),
        scaled_positions=ase_cell.get_scaled_positions(),
        symbols=ase_cell.get_chemical_symbols()
    )
    
    ph_mlip = phonopy.Phonopy(ph_atoms, 
        supercell_matrix=ph_ref.supercell_matrix,
        primitive_matrix=ph_ref.primitive_matrix
    )

    # Generate displacements and calculate forces in the supercells
    forcesets = []
    ph_mlip.generate_displacements(distance=distance, is_diagonal=False)
    
    for supercell in ph_mlip.supercells_with_displacements:
        scell = Atoms(
            cell=supercell.cell,
            symbols=supercell.symbols,
            scaled_positions=supercell.scaled_positions,
            pbc=True)
        scell.calc = mlip.calculator
        
        forces = scell.get_forces()
        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]

        forcesets.append(forces)

    ph_mlip.forces = forcesets
    ph_mlip.produce_force_constants()
    ph_mlip.symmetrize_force_constants()

    return ph_mlip, relaxed


def calculate_band_structure(ph, structure_name, output_dir="band_structure"):
    """Compute and plot the phonon band structure along high-symmetry paths, only plotting continuous path segments with labels"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use seekpath to determine the high-symmetry path
    bands, labels, path_connections = get_band_qpoints_by_seekpath(
        ph.primitive, 
        npoints=101, 
        is_const_interval=True
    )
    
    print(f"High-symmetry path labels: {labels}")
    print(f"Path connections: {path_connections}")
    
    # Compute band structure
    ph.run_band_structure(
        bands,
        path_connections=path_connections,
        labels=labels,
        is_legacy_plot=False
    )
    
    band_dict = ph.get_band_structure_dict()
    

    return band_dict


from argparse import ArgumentParser



def get_params():
    parser = ArgumentParser(description="Compute and plot phonon band structures")
    parser.add_argument('--model', type=str, default="mattersim-v1", help='Model to use')
    parser.add_argument('--ref', type=str, default='metal-mp50-gen/', help='Folder containing reference YAML files')
    parser.add_argument('--dest', type=str, default="metal-mp50-gen-Phono-mattersim-v1/", help='Output folder')
    
    parser.add_argument('--distance', type=float, default=0.01, help='Displacement distance')
    parser.add_argument('--relax', default=True, help='Whether to relax structure')
    parser.add_argument('--relaxedDest', type=str, default="metal-mp50-gen-Phono-mattersim-v1/", help='Output folder')
    
    # 修改参数：现在支持更细粒度的划分
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index (0-3)')
    parser.add_argument('--subpart_index', type=int, default=0, help='Sub-part index within GPU (0-2)')
    parser.add_argument('--total_gpus', type=int, default=4, help='Total number of GPUs')
    parser.add_argument('--subparts_per_gpu', type=int, default=3, help='Number of sub-parts per GPU')

    if 'ipykernel' in sys.modules:
        return parser.parse_args([])
    else:
        return parser.parse_args()

params = get_params()
print(params)

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu_index)
print(f"Using GPU: {params.gpu_index}, Sub-part: {params.subpart_index}")

# Statistics
success_count = 0
imag_count = 0
no_imag_count = 0
no_imag_files = []
failed_items = []
Lable = []

# Initialize ML potential
mlip = umlip(model=params.model)

# Find YAML files
yml_files = glob.glob(f"{params.ref}/*.yaml*")
if not yml_files:
    print(f"No YAML files found in {params.ref}")
    exit()

yml_files = sorted(yml_files)

# 计算更细粒度的文件划分
total_files = len(yml_files)
total_subparts = params.total_gpus * params.subparts_per_gpu
subpart_size = total_files // total_subparts

# 计算当前子部分的文件范围
subpart_index = params.gpu_index * params.subparts_per_gpu + params.subpart_index
start_idx = subpart_index * subpart_size

if subpart_index == total_subparts - 1:
    # 最后一个子部分处理剩余的所有文件
    end_idx = total_files
else:
    end_idx = start_idx + subpart_size

part_files = yml_files[start_idx:end_idx]
print(f"Processing GPU {params.gpu_index}, sub-part {params.subpart_index} (global subpart {subpart_index}): files {start_idx} to {end_idx-1} ({len(part_files)} files)")

output_dir = params.dest
relaxed_output_dir = params.relaxedDest
Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(relaxed_output_dir).mkdir(parents=True, exist_ok=True)

# 为每个子部分创建独立的输出目录
subpart_output_dir = os.path.join(output_dir, f"gpu{params.gpu_index}_part{params.subpart_index}")
subpart_relaxed_dir = os.path.join(relaxed_output_dir, f"gpu{params.gpu_index}_part{params.subpart_index}")
Path(subpart_output_dir).mkdir(parents=True, exist_ok=True)
Path(subpart_relaxed_dir).mkdir(parents=True, exist_ok=True)

# 检查该子部分是否已经有标签文件，如果有则加载
subpart_label_file = os.path.join(subpart_relaxed_dir, "Lable.txt")
existing_labels = {}
if os.path.exists(subpart_label_file):
    print(f"Loading existing labels from: {subpart_label_file}")
    with open(subpart_label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                formula, status = line.split('\t')
                existing_labels[formula] = status
    print(f"Loaded {len(existing_labels)} existing labels")
else:
    print(f"No existing label file found at: {subpart_label_file}")

# 初始化标签列表，先包含已有的标签
for formula, status in existing_labels.items():
    Lable.append([formula, status])
    if status == 'Stable':
        no_imag_files.append(formula)

summary_file = os.path.join(subpart_output_dir, "1Top-summary.txt")
count_num = 0
processed_count = 0
skipped_count = 0
sorted_part_files = sorted(part_files)

for yml_file in sorted_part_files:
    count_num += 1
    filename = os.path.basename(yml_file).split('.')[0]
    structure_name = filename
    
    # 检查是否已经在标签文件中
    if structure_name in existing_labels:
        print(f"\nSkipping {structure_name}: already in label file (status: {existing_labels[structure_name]})")
        skipped_count += 1
        continue
    
    print(f"\nProcessing file: {yml_file}")
    
    try:
        ph_ref = phonopy.load(yml_file)
        
        result = run_phonopy(ph_ref, mlip, distance=params.distance, relax=params.relax)
        if result is None:
            Lable.append([structure_name,'unStable'])
            print("Error: relaxation failed")
            continue
        
        ph_mlip, relaxed = result
        # 保存弛豫后的结构
        if params.relax and relaxed[0] is not None:
            cif_filename = os.path.join(subpart_relaxed_dir, f"{structure_name}_relaxed.cif")
            relaxed[0].write(cif_filename)
            print(f"Relaxed structure saved: {cif_filename}")
        
        fc_filename = os.path.join(subpart_output_dir, f"{structure_name}.yaml")
        ph_mlip.save(filename=fc_filename, settings={'force_constants': True})
        print(f"Force constants saved: {fc_filename}")
        
        band_data = calculate_band_structure(ph_mlip, structure_name, subpart_output_dir)
        
        all_freqs = np.concatenate(band_data['frequencies'])
        if np.any(all_freqs < -1e-3):
            imag_count += 1
            Lable.append([structure_name,'unStable'])
        else:
            no_imag_count += 1
            no_imag_files.append(structure_name)
            Lable.append([structure_name,'Stable'])
        
        success_count += 1
        processed_count += 1

    except Exception as e:
        print(f"Error while processing {yml_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        failed_items.append((structure_name, str(e)))
        Lable.append([structure_name,'unStable'])
        continue
        
    if count_num % 10 == 0:
        # 保存部分标签文件
        info_file = os.path.join(subpart_relaxed_dir, "Lable.txt")
        with open(info_file, 'w') as f:
            # 按结构名排序后写入
            sorted_labels = sorted(Lable, key=lambda x: x[0])
            for info in sorted_labels:
                f.write(f"{info[0]}\t{info[1]}\n")
        print(f"Saved partial label file with {len(sorted_labels)} entries")

# 保存完整的标签文件（按结构名排序）
info_file = os.path.join(subpart_relaxed_dir, "Lable.txt")
with open(info_file, 'w') as f:
    sorted_labels = sorted(Lable, key=lambda x: x[0])
    for info in sorted_labels:
        f.write(f"{info[0]}\t{info[1]}\n")
print(f"Saved complete label file with {len(sorted_labels)} entries to: {info_file}")

# Write summary file
with open(summary_file, 'w') as f:
    f.write(f"GPU {params.gpu_index}, Sub-part {params.subpart_index} Summary:\n")
    f.write(f"Total files in this partition: {len(part_files)}\n")
    f.write(f"Files already in label file and skipped: {skipped_count}\n")
    f.write(f"Files newly computed: {processed_count}\n")
    f.write(f"Total successful calculations (including existing): {len(Lable)}\n")
    f.write(f"Files with imaginary modes: {imag_count}\n")
    f.write(f"Files without imaginary modes: {no_imag_count}\n\n")

    f.write("List of dynamically stable structures:\n")
    for name in no_imag_files:
        f.write(f"{name}\n")

    f.write("\nFailed structures and reasons:\n")
    for name, reason in failed_items:
        f.write(f"{name}: {reason}\n")

print(f"\nGPU {params.gpu_index}, Sub-part {params.subpart_index} summary:")
print(f"  Files in partition: {len(part_files)}")
print(f"  Skipped (already in label file): {skipped_count}")
print(f"  Newly computed: {processed_count}")
print(f"  Total entries in final label file: {len(Lable)}")
print(f"  Dynamically stable: {no_imag_count}")
print(f"  Summary saved to: {summary_file}")
print(f"  Complete label file saved to: {info_file}")
print(f"GPU {params.gpu_index}, Sub-part {params.subpart_index} calculation completed!")
