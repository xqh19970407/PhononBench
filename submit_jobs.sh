#!/bin/bash

# 物理 GPU
phys_gpus=(0 1 2 3)

# Python 中额外使用的逻辑 GPU 0–3
logic_gpus=(0 1 2 3)

subparts_per_gpu=3

for i in "${!phys_gpus[@]}"; do
    phys_gpu=${phys_gpus[$i]}
    logic_gpu=${logic_gpus[$i]}

    for subpart_index in $(seq 0 $((subparts_per_gpu-1)))
    do
        global_index=$((logic_gpu * subparts_per_gpu + subpart_index))

        echo "Submitting job: physical GPU $phys_gpu, logic GPU $logic_gpu, sub-part $subpart_index"

        CUDA_VISIBLE_DEVICES=$phys_gpu \
        nohup python -u phonon_multi_gpu_run.py \
            --ref /home/xqhan/InvDesFlow3.0/Benchmark/MatterGen-gen/dft_band_gap/bg_3.5/phonon-calculation-input \
            --dest /home/xqhan/InvDesFlow3.0/Benchmark/MatterGen-gen/dft_band_gap/bg_3.5/phonon-calculation-output/ \
            --relaxedDest /home/xqhan/InvDesFlow3.0/Benchmark/MatterGen-gen/dft_band_gap/bg_3.5/relaxed/ \
            --model=mattersim-v1 \
            --gpu_index $logic_gpu \
            --subpart_index $subpart_index \
            --total_gpus 4 \
            --subparts_per_gpu $subparts_per_gpu \
            > /home/xqhan/InvDesFlow3.0/Benchmark/MatterGen-gen/dft_band_gap/bg_3.5/phonon-gpu${phys_gpu}-part${subpart_index}.log 2>&1 &

        echo "Job started with PID: $!"
        sleep 5
    done
done

echo "All jobs submitted!"
