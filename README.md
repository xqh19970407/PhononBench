# PhononBench
PhononBench is a phonon-based benchmark for large-scale dynamical stability evaluation of AI-generated crystals, featuring 100k+ structures, DFT-level MatterSim phonon calculations, and open-source high-throughput workflows.

## Installation

PhononBench relies on **MatterSim** for DFT-level phonon calculations.
We therefore **strongly recommend installing MatterSim first**, following the official environment setup, before using PhononBench.

### Prerequisites

* **Python ≥ 3.10**
* **mamba or micromamba** (recommended for fast and reliable dependency resolution)
* Linux environment (recommended for large-scale phonon calculations)

We recommend installing MatterSim **from source using mamba**, as this is the most reliable setup for large-scale phonon calculations and was the environment used in this work.

```bash
# clone MatterSim
git clone https://github.com/microsoft/mattersim.git
cd mattersim

# create the environment
mamba env create -f environment.yaml
mamba activate mattersim

# install MatterSim in editable mode
uv pip install -e .
```

## Citation

If you use **PhononBench** in your research, please cite the following paper:

```bibtex
@misc{han2025phononbenchalargescalephononbasedbenchmark,
  title        = {PhononBench: A Large-Scale Phonon-Based Benchmark for Dynamical Stability in Crystal Generation},
  author       = {Xiao-Qi Han and Ze-Feng Gao and Peng-Jie Guo and Zhong-Yi Lu},
  year         = {2025},
  eprint       = {2512.21227},
  archivePrefix= {arXiv},
  primaryClass = {cond-mat.mtrl-sci},
  url          = {https://arxiv.org/abs/2512.21227}
}
```
