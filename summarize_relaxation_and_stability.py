import os
from pathlib import Path

# 需要处理的模型（名称: 路径）
model_paths = {
    "CrystalFlow-gen": "/path/PhononBench/CrystalFlow-gen/relaxed",
    "CrystalFormer-alex20": "/path/PhononBench//CrystalFormer-gen/alex20/relaxed",
    "CrystalFormer-mp20": "/path/PhononBench//CrystalFormer-gen/mp20/relaxed",
    "CrystalLLM-gen": "/path/PhononBench//CrystalLLM-gen/relaxed",
    "DiffCSP-gen": "/path/PhononBench//DiffCSP-gen/relaxed",
    "InvDesFlow-AL-DPO": "/path/PhononBench//InvDesFlow-AL/DPO/relaxed",
    "MatterGen-base": "/path/PhononBench//MatterGen-gen/mattergen_base/relaxed",
    "MatterGen-mp20-base": "/path/PhononBench//MatterGen-gen/mp_20_base/relaxed",
}

def read_label_file(file_path):
    """读取单个 Lable.txt，返回列表 [('formula', 'Stable'), ...]"""
    items = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
    return items

def collect_model_results(model_dir):
    """在模型目录下递归查找所有 Lable.txt 并统计"""
    all_items = []

    for root, dirs, files in os.walk(model_dir):
        if "Lable.txt" in files:
            file_path = os.path.join(root, "Lable.txt")
            all_items.extend(read_label_file(file_path))

    return all_items

def count_relaxed_cifs(model_dir):
    """统计每个模型下 _relaxed.cif 的数量"""
    count = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith("_relaxed.cif"):
                count += 1
    return count


print("\n==========  Stability & Relaxation Statistics  ==========\n")

for model_name, model_dir in model_paths.items():

    # 1. 统计 relaxed CIF 的数量
    cif_total = count_relaxed_cifs(model_dir)

    # 2. 统计 Lable.txt 数据
    items = collect_model_results(model_dir)

    if not items:
        print(f"{model_name}: No Lable.txt found!  (Relaxed CIFs = {cif_total})")
        continue

    stable = sum(1 for x in items if x[1].lower() == "stable")
    unstable = sum(1 for x in items if x[1].lower() == "unstable")
    total_labeled = stable + unstable

    # 避免除零
    ratio_stable = stable / total_labeled * 100 if total_labeled > 0 else 0
    relax_success_ratio = cif_total / total_labeled * 100 if cif_total > 0 else 0
    stable_from_total_ratio = stable / cif_total * 100 if cif_total > 0 else 0

    print(f"{model_name}:")
    print(f"  Total CIFs Submitted        = {total_labeled}")
    print(f"  Relaxed Successfully  = {cif_total}")
    print(f"  Stable                      = {stable}")
    print(f"  UnStable                    = {unstable}")
    print(f"  Stable From Total Ratio = {ratio_stable:.2f}%")
    print(f"  Relaxation Success Ratio    = {relax_success_ratio:.2f}%")
    print(f"  Stable Ratio (relaxed only)     = {stable_from_total_ratio:.2f}%")
    print("")

print("==========================================================\n")


# -------- 追加：最终输出格式 --------
final_data = {}

for model_name, model_dir in model_paths.items():
    # 统计 CIF 数
    cif_total = count_relaxed_cifs(model_dir)

    # 统计标签
    items = collect_model_results(model_dir)
    stable = sum(1 for x in items if x[1].lower() == "stable")
    unstable = sum(1 for x in items if x[1].lower() == "unstable")
    total_labeled = stable + unstable

    # 保存数据
    final_data[model_name] = (cif_total, stable)

print("\n==========  Final Summary Dict  ==========\n")
print("data = {")
for k, v in final_data.items():
    print(f'    "{k}": {v},')
print("}\n")
print("==========================================\n")

