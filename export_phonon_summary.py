#!/usr/bin/env python3
"""
Export phonon minimum frequencies and stability summary.

Usage:
python export_phonon_summary.py \
    --phonon_dir /your_path/phonon-calculation-output \
    --relaxed_dir /your_path/relaxed \
    --output_csv phonon_summary.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import phonopy

from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from argparse import ArgumentParser
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath


# =========================
# 限制线程，避免 CPU 爆炸
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# =========================
# 参数
# =========================
def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--phonon_dir",
        type=str,
        required=True,
        help="Directory containing phonon force constant yaml files"
    )

    parser.add_argument(
        "--relaxed_dir",
        type=str,
        default=None,
        help="Directory containing relaxed cif files"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="phonon_summary.csv",
        help="Output csv filename"
    )

    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Save band structure npz files"
    )

    parser.add_argument(
        "--npz_dir",
        type=str,
        default="phonon_npz",
        help="Directory for saving npz files"
    )

    parser.add_argument(
        "--imag_threshold",
        type=float,
        default=-1e-3,
        help="Imaginary frequency threshold"
    )

    return parser.parse_args()


# =========================
# 单结构处理
# =========================
def process_single(task):
    yaml_file, relaxed_dir, save_npz, npz_dir, imag_threshold = task

    structure_name = os.path.basename(yaml_file).replace(".yaml", "")

    try:
        ph = phonopy.load(yaml_file)

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

        frequencies = band_dict["frequencies"]

        all_freqs = np.concatenate(frequencies)

        min_freq = float(np.min(all_freqs))

        is_stable = bool(min_freq >= imag_threshold)

        # 保存 npz
        npz_path = None

        if save_npz:
            Path(npz_dir).mkdir(parents=True, exist_ok=True)

            npz_path = os.path.join(
                npz_dir,
                f"{structure_name}.npz"
            )

            np.savez(
                npz_path,
                frequencies=np.array(
                    frequencies,
                    dtype=object
                ),
                labels=np.array(labels, dtype=object),
                min_frequency=min_freq
            )

        # 尝试寻找 relaxed cif
        cif_path = None

        if relaxed_dir is not None:

            cif_candidates = glob.glob(
                f"{relaxed_dir}/**/{structure_name}_relaxed.cif",
                recursive=True
            )

            if len(cif_candidates) > 0:
                cif_path = cif_candidates[0]

        return {
            "structure_name": structure_name,
            "min_frequency": min_freq,
            "is_stable": is_stable,
            "yaml_path": yaml_file,
            "cif_path": cif_path,
            "npz_path": npz_path,
        }

    except Exception as e:

        return {
            "structure_name": structure_name,
            "min_frequency": None,
            "is_stable": False,
            "yaml_path": yaml_file,
            "cif_path": None,
            "npz_path": None,
            "error": str(e)
        }


# =========================
# 主函数
# =========================
def main():

    args = get_args()

    yaml_files = glob.glob(
        f"{args.phonon_dir}/**/*.yaml",
        recursive=True
    )

    yaml_files = sorted(yaml_files)

    print(f"Found yaml files: {len(yaml_files)}")

    if len(yaml_files) == 0:
        print("No yaml files found.")
        return

    tasks = []

    for y in yaml_files:

        tasks.append((
            y,
            args.relaxed_dir,
            args.save_npz,
            args.npz_dir,
            args.imag_threshold
        ))

    ncpu = max(cpu_count() // 2, 1)

    print(f"Using CPUs: {ncpu}")

    with Pool(ncpu) as pool:

        results = list(tqdm(
            pool.imap_unordered(process_single, tasks),
            total=len(tasks)
        ))

    df = pd.DataFrame(results)

    # 排序
    df = df.sort_values("structure_name")

    # 保存 CSV
    df.to_csv(args.output_csv, index=False)

    print("\nFinished.")
    print(f"CSV saved to: {args.output_csv}")

    stable_num = int(df["is_stable"].sum())

    print(f"Stable structures: {stable_num}")
    print(f"Unstable structures: {len(df) - stable_num}")


if __name__ == "__main__":
    main()
