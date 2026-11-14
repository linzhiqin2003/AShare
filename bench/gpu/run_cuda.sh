#!/usr/bin/env bash
# 在一台 NVIDIA Linux 机器上一条命令复现 GPU 探针（合成数据，与真实面板同规模 2430 × 5923）。
#
#   bash bench/gpu/run_cuda.sh            # 在仓库根目录运行
#
# 前提：Linux x86_64 或 aarch64、glibc >= 2.35、NVIDIA 驱动 >= 550（MLX 的 CUDA 12 包要求；
# CUDA 13 包要驱动 >= 580）、算力 >= SM 7.5、Rust stable、uv。结果写到 bench/gpu/results/。
set -euo pipefail
cd "$(dirname "$0")/../.."

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv sync --all-packages --group dev --group gpu-probe-cuda --inexact
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'torch 看不到 CUDA'; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
uv run --no-sync python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device()); mx.eval(mx.ones(4) + 1)"

# 1. 算子差分检查（GPU float32 对 panelops float64）；mlx-fused 在 Linux 上自动改用 CUDA 内核正文
uv run --no-sync python bench/gpu/check_ops.py --backend torch --device cuda --synthetic --t 600 --n 800
uv run --no-sync python bench/gpu/check_ops.py --backend mlx --synthetic --t 600 --n 800
uv run --no-sync python bench/gpu/check_ops.py --backend mlx-fused --synthetic --t 600 --n 800
# 2. CPU 现行做法与整代共享（生成种群和 float64 参考，约几分钟）
uv run --no-sync python bench/gpu/cpu_profile.py --synthetic --threads "${THREADS:-8}"
# 3. 整代粗筛：框架算子写法（torch、mlx）与自定义内核写法（mlx-fused）
uv run --no-sync python bench/gpu/gpu_probe.py --backend torch --device cuda --synthetic --repeat 3
uv run --no-sync python bench/gpu/gpu_probe.py --backend mlx --synthetic --repeat 3
uv run --no-sync python bench/gpu/gpu_probe.py --backend mlx-fused --synthetic --repeat 5
ls -l bench/gpu/results/
