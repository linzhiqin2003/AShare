#!/usr/bin/env bash
# 在一台 NVIDIA Linux 机器上一条命令验证 GPU 粗筛（ashare.factor.mining.gpu）的 CUDA 路径。
#
#   bash bench/gpu/verify_cuda.sh          # 仓库根目录；结果写到 bench/gpu/results/verify_cuda.json
#
# 前提（docs/GPU_MINING.md「CUDA」）：Linux x86_64 或 aarch64、glibc >= 2.35、NVIDIA 驱动 >= 550
# （mlx[cuda12]）、算力 >= SM 7.5、Rust stable、uv。没有真实数据也能跑（合成面板，与真实面板
# 同规模 2430 × 5923）；有数据时设 ASHARE_DATA_DIR 再加 REAL=1 另测真实面板。
#
# 依次：记录机器信息；装依赖（含 Rust 扩展）；确认 MLX 在 GPU 上、内核能编译；跑 GPU 测试
# （算子差分、整代粗筛、选择不变性、确定性；CPU 冻结摘要只在生成它的平台上比，这里会 skip）；
# 真实规模计时（256、1024 个表达式一代）。任何一步失败都会写进 JSON 再以非零退出。
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=bench/gpu/results
mkdir -p "$OUT"
LOG="$OUT/verify_cuda.log"
: > "$LOG"
status=0

step() {  # step 名字 命令...
  local name=$1; shift
  echo "== $name" | tee -a "$LOG"
  if "$@" >> "$LOG" 2>&1; then echo "ok" > "$OUT/.step_$name"; else echo "failed" > "$OUT/.step_$name"; status=1; fi
}

step machine bash -c 'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv; ldd --version | head -1; uname -m'
step sync uv sync --all-packages --group dev --extra gpu --inexact
step probe uv run --no-sync python -c "
from ashare.factor.mining.gpu import available
from ashare.factor.mining.gpu.backend import MLXBackend
ok, why = available()
assert ok, why
b = MLXBackend()
print('device', b.device, 'kernel', b.kernel_id)
"
step tests uv run --no-sync pytest test/factor/gpu -q -p no:randomly -n 2 \
  --junitxml="$OUT/verify_cuda_junit.xml"
step timing uv run --no-sync python bench/gpu/screen_profile.py --synthetic --sizes 256,1024 --repeat 3
if [ "${REAL:-0}" = "1" ]; then
  step timing_real uv run --no-sync python bench/gpu/screen_profile.py --sizes 256,1024 --repeat 3
fi

uv run --no-sync python - "$OUT" <<'PY'
import json, platform, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path

out = Path(sys.argv[1])
steps = {p.name[6:]: p.read_text().strip() for p in sorted(out.glob(".step_*"))}
for p in out.glob(".step_*"):
    p.unlink()
report = {"steps": steps, "machine": platform.platform()}
try:
    report["gpu"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except OSError:
    report["gpu"] = None
junit = out / "verify_cuda_junit.xml"
if junit.exists():
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    report["tests"] = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    report["failed_tests"] = [
        f"{c.get('classname')}::{c.get('name')}" for c in suite.iter("testcase")
        if c.find("failure") is not None or c.find("error") is not None
    ]
for name in sorted(out.glob("screen_profile_cuda_*.json")):
    blob = json.loads(name.read_text())
    report[name.stem] = {
        "peak_bandwidth_gbs": blob["peak_bandwidth"] / 1e9,
        "sizes": {k: {"median_s": v["median"], "per_expr_ms": v["per_expr_ms"],
                      "lower_bound_s": v["lower_bound_seconds"], "routed": v["routed"]}
                  for k, v in blob["sizes"].items()},
        "stages": blob["stages"]["seconds"],
    }
(out / "verify_cuda.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
print(json.dumps(report, ensure_ascii=False, indent=1))
PY
exit $status
