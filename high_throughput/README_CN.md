# 与 PhononBench 网页条件对齐的高通量计算

2026-11-09 更新。保留仓库原有论文复现脚本，新增独立、可移植的 CIF → 声子入口。使用网页当前的 MatterSim 5M 权重，支持 3D / 2D、无弛豫、固定晶胞弛豫、定压晶胞弛豫、保持对称性、批量任务和多 GPU。

这是离线计算工具，不连接网站或写入网站用户记录。无网页的原子数、时段和提交频率限制；每个材料实际计算仍限制为 **20 分钟**。模型结果属于机器学习势下的谐近似声子计算，并非重新执行 DFT。

## 1. 下载与环境

完整包包含当前模型、计算代码、MatterSim 推理源码、环境 TXT、Linux / Python 3.10 的辅助 wheel 及说明：

- [完整 ZIP](https://media.githubusercontent.com/media/xqh19970407/PhononBench/main/distributions/PhononBench_high_throughput_20261109.zip)
- [ZIP 校验值](../distributions/PhononBench_high_throughput_20261109.zip.sha256)
- [模型校验与许可](../models/manifest.json)

Git 仓库的权重和 ZIP 使用 Git LFS；直接下载上述完整 ZIP 得到真实权重，不需要 Git LFS。通过 git clone 使用时先执行 `git lfs pull`，不要把 LFS 指针当作模型。

已验证的平台为 Linux x86_64、Python 3.10.19。环境完整版本清单在 [environment](../environment)：

| 文件 | 用途 |
|---|---|
| `requirements-runtime.txt` | 74 个推理 / 声子 / 绘图依赖的固定版本，Torch 单独安装 |
| `pip-freeze-current.txt` | 网页当前整个环境的已安装分发包快照，供审计；不是最小安装清单 |
| `conda-explicit-linux-64.txt` | 网页环境 Conda 层的显式导出，不包含全部 pip 包 |
| `conda-packages.json`、`python-version.txt` | 更完整的环境记录 |

包内没有复制整个数 GB 的 Conda 目录。已有网页 `mattersim` 环境可直接运行；新机器建议建立独立环境，以免改变其他服务。

```bash
cd PhononBench
python3.10 -m venv .venv
source .venv/bin/activate
# GPU 环境；需要适配 CUDA 12.8 的 NVIDIA 驱动。
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
# CPU 用户可改用：https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -r environment/requirements-runtime.txt
python -m pip install --no-deps vendor/wheels/mattersim-0.0-cp310-cp310-linux_x86_64.whl
```

`requirements-runtime.txt` 已包含原版 Phonopy 2.47.1，也可用包内对应 wheel。二维路径实现在本项目中，不要求改写 site-packages 的 Phonopy。

不适用预编译 MatterSim wheel 时，从推理源码编译，需要 C 编译器：

```bash
python -m pip install --no-deps --no-build-isolation ./vendor/MatterSim
```

推理 Python/Cython 源码保持当前环境版本；只精简了安装元数据与依赖，原上游配置另存为 `pyproject.upstream.toml`。本机安装元数据版本为 `mattersim==0.0`，不应据此误认成某个新的上游发行版。源码哈希、MIT 许可及 Model Card 随包保留。

## 2. 单材料：网页参数映射

| 网页选项 | 命令行 |
|---|---|
| 三维材料 | `--dimension 3d`，默认超胞 `2 2 2` |
| 二维材料 | `--dimension 2d`，默认超胞 `4 4 1`，能带和 DOS 网格限定 qz=0 |
| 不弛豫 | `--relax-type none` |
| NVT | `--relax-type nvt`，固定晶胞，只优化原子位置 |
| NPT | `--relax-type npt --pressure 0`，优化原子和晶胞 |
| 高压 | `--relax-type npt --pressure 10`，正值表示压缩，单位 GPa |
| 保持对称性 | `--fix-symmetry`，默认开启；关闭用 `--no-fix-symmetry` |

NVT / NPT 在这里是网页沿用的静态优化名称，不是有限温度分子动力学。默认位移 0.01 Å，Phonopy 晶体对称约化开启，正负位移为 auto，FIRE 的优化目标为 0.0025 eV/Å，对应网页传入 fmax=0.005 后采用半阈值的规则。

```bash
# 3D、2 GPa、保持对称性；默认 CPU 两线程。
python high_throughput/phononbench.py \
  --cif high_throughput/examples/Si.cif --out results/Si_2GPa \
  --dimension 3d --relax-type npt --pressure 2

# 2D、固定晶胞；显式使用某张 GPU。
CUDA_VISIBLE_DEVICES=0 python high_throughput/phononbench.py \
  --cif high_throughput/examples/graphene.cif --out results/graphene \
  --dimension 2d --relax-type nvt --device cuda

# 不弛豫，显式设置超胞、网格并关闭对称约化。
python high_throughput/phononbench.py \
  --cif input.cif --out results/input \
  --relax-type none --dim 2 2 2 --mesh 8 8 8 \
  --no-fix-symmetry --no-use-symmetry
```

默认只使用当前可用的前两个逻辑 CPU；使用共享服务器时，显式指定有余量的 `--cpu-cores 8,9 --threads 2`。CPU 编号由实际机器决定。`--preflight` 只检查 CIF、实际对称性、超胞规模和位移数，不加载模型或计算受力。

## 3. 二维与网页实现的区别

当前网页按 `dim[-1]==1` 选择二维能带，并依赖本机的 Phonopy 修改。新版通过明确的 `--dimension` 选择，不会把一个使用 `1 1 1` 超胞的三维体系误判为二维；二维能带路径与网页采用相同的面内晶格分类和几何坐标。

二维 NPT 默认 `--cell-policy slab`：只放开面内晶胞自由度，保留真空方向，要求 ab 在 xy 平面、c 垂直于该平面。复现当前网页的全三维晶胞弛豫行为时，显式添加 `--cell-policy web`。

含真空的二维晶胞不能直接把三维 GPa 当作不依赖真空厚度的二维压力。默认 slab 模式拒绝二维非零压力；只有显式选择 `--cell-policy web` 才按网页的三维晶胞压力约定计算。做受压二维材料研究时，应先确定物理边界条件和压力定义。

二维 DOS 默认 Gaussian 展宽 0.05 THz；三维默认原版 Phonopy 的 tetrahedron DOS。均可用 `--dos-method` 选择。DOS 频率积分步长为 `--frequency-pitch 0.01` THz，避免窄 Gaussian 峰因自动步长过粗而漏采样。默认网格密度与网页相同，为 `--mesh-density 100`；`--mesh NX NY NZ` 可覆盖，二维强制 NZ=1。`--no-pdos` 可节省特征向量内存。输出文件和网格策略有记录，参数对齐不表示每个文件逐字节相同。

## 4. 高通量：同条件扫描或混合任务

同一目录、多个压力：

```bash
python high_throughput/run_batch.py \
  --input-dir my_cifs --out results/pressure_scan \
  --dimension 3d --relax-type npt --pressures 0 5 10 \
  --workers 2 --threads 2 --cpus 8,9,10,11 --gpus 0,1
```

混合 2D / 3D、压力和弛豫方式，使用 [examples/jobs.json](examples/jobs.json)。每个 `jobs` 条目指定唯一 `id`、CIF 路径及独立 `args`；CIF 相对清单所在目录解析：

```bash
python high_throughput/run_batch.py \
  --manifest high_throughput/examples/jobs.json --out results/mixed \
  --workers 2 --threads 2 --cpus 8,9,10,11 --gpus 0,1
```

不传 `--gpus` 就用 CPU；传入时每个 worker 对应一张 GPU，不自动在同一 GPU 上堆叠 worker。`workers × threads` 必须落在选定 CPU 预算内，默认可选预算为可用逻辑核的一半。多个批次和其他服务的总预算仍需一起核算。

`--plan-only` 导出计划而不计算。完成记录写到 `batch_results.json/.csv`，单材料结果放在以 `id` 命名的目录。再次运行同一批次时，只跳过已 DONE 且输入、参数、代码、环境版本、权重和 ZIP 哈希均匹配的任务；不把目录存在当作完成。失败或部分任务需审查后显式加 `--resume`，不自动重新提交失败任务。

## 5. 超时、断点及结果

每个材料从实际启动开始最长 1200 秒，含模型初始化、弛豫、位移、后处理和打包，排队另计。可用 `--timeout-seconds` 缩短，不能超过 1200。超时停止整个计算子进程组，先 SIGTERM，最多再等 5 秒后 SIGKILL，返回 124 并标记 TIMEOUT，无自动重试。

Linux worker 另设独立计时器与父进程死亡保护，监督进程被意外终止时不会留下持续计算的 worker。异常中断后的状态文件可能保留最后现场，检查后再显式续算。

已完成的结构弛豫和逐位移受力保存在输出目录，单材料显式 `--resume` 或批次 `--resume` 可复用匹配断点，每次仍受 20 分钟限制。输入、模型、计算代码或参数不匹配会拒绝复用；未完成的优化器状态不自动恢复。可选 `--temperature-limit 68` 在有 CPU 传感器时暂停降温，等待也计入时限。

每个材料输出：

- `summary.json/.txt`：条件、能量 / 焓、残余力 / 应力、频率、虚频比例与版本信息。
- `input.cif`、`relaxed.cif`、`relaxation.json`：输入和实际用于声子的结构；不弛豫时也会明确标记 none。
- `phonopy_fc.yaml`：结构、位移、受力及力常数，可用原版 Phonopy 重读。
- `phonon_band.png/.pdf/.dat`、`phononbench_band.yaml`：能带与原始数据。
- `total_dos.dat`、`phonon_dos.png`、`projected_dos.dat/.png`：总与投影 DOS。
- `mesh_frequencies.npz`、`forces/`、`events.jsonl`、`worker.log`：网格频率、断点及日志。
- `PhononBench-results.zip`：可分发结果包，排除内部长轨迹与全部受力断点。

频率取负表示虚频；小 Γ 声学负值、二维柔性声学支和有限超胞效应需要结合收敛情况解释。结果不直接证明有限温度或热力学稳定性。

## 6. 复现与许可

本更新保留根目录历史脚本，原论文数据及其标签不因新增入口而改变。部署源码基线哈希见 [source_provenance.json](source_provenance.json)，实际验证见 [VALIDATION_CN.md](VALIDATION_CN.md)。

模型和 MatterSim 推理源码沿用 Microsoft 的 MIT 许可，见 [LICENSE.txt](../vendor/MatterSim/LICENSE.txt) 与 Model Card。Phonopy 使用 BSD 许可，随附 wheel 保留其许可。第三方声明见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
