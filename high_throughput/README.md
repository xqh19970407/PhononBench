# Portable high-throughput PhononBench

The new entry points support 3D and 2D phonons, no relaxation, fixed-cell
relaxation (NVT), variable-cell relaxation (NPT), pressure in GPa, optional
symmetry constraints, bounded CPU/multi-GPU batches, force checkpoints and a
20-minute deadline per material. Legacy benchmark scripts remain unchanged.

See [the complete guide](README_CN.md) for installation, manifests, outputs and
the precise correspondence with the current website.

Download the [complete code/model ZIP](https://media.githubusercontent.com/media/xqh19970407/PhononBench/main/distributions/PhononBench_high_throughput_20261109.zip).
It includes the website's MatterSim 5M checkpoint, inference source, pinned
environment TXT files, examples and supporting wheels for Linux/Python 3.10.
Dependencies still need to be installed; this is not a copied Conda environment.
When cloning instead, install Git LFS and run `git lfs pull` for model/ZIP assets.

```bash
python high_throughput/phononbench.py --cif input.cif --out results/sample \
  --dimension 3d --relax-type npt --pressure 5

CUDA_VISIBLE_DEVICES=0 python high_throughput/phononbench.py \
  --cif slab.cif --out results/slab --dimension 2d --relax-type nvt --device cuda

python high_throughput/run_batch.py --manifest high_throughput/examples/jobs.json \
  --out results/batch --workers 2 --threads 2 --cpus 0,1,2,3 --gpus 0,1
```

NVT/NPT denote static relaxation modes, not finite-temperature MD. Positive
pressure is compressive. The default slab cell policy preserves the vacuum
direction for 2D NPT relaxation; `--cell-policy web` explicitly reproduces the
website's full-cell convention. Nonzero 3D pressure in a vacuum slab requires
that explicit convention and depends on the vacuum thickness. All 2D q-points
have qz=0; no patched Phonopy installation is required.

Completed batch jobs are skipped only after matching request/model/code/version
and result ZIP hashes. Failed jobs require explicit `--resume`; no automatic
retry or deadline extension occurs. The workflow uses a machine-learning
potential and does not execute DFT or establish finite-temperature stability.
