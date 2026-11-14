"""``mine(screen=...)``：CPU 粗筛与改动前逐位一致；GPU 粗筛 + CPU 复算后的选择与纯 CPU 完全相同。

来源：GPU 粗筛任务书（2026-11-14）与 docs/GPU_PROBE.md 第六节：

- ``screen="cpu"``（默认）的全部可观察输出与加入 GPU 粗筛之前逐位相同：冻结摘要
  ``fixtures/mining_cpu_digest.json`` 由改动前的代码（main 0482f28）生成；
- 选择不变性：合成面板上 GPU 粗筛 + CPU 复算的回测候选、适应度、入选、汇总（含多重检验
  计数）、指纹与收益序列文件，与纯 CPU 路径完全一致；GPU 判为不过线的在纯 CPU 路径上都是
  粗筛掉的、且训练段 IC 不显著；
- 台账里没有 float32 的数字；GPU 记录复用台账时按 CPU 重算；
- GPU 不可用时报错，不退回 CPU。

没装 MLX 或没有 GPU 时，只跑 CPU 的部分。
"""

import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import gpu_cases as gc
import pytest

from ashare.factor.mining import gpu as gpumod
from ashare.factor.mining.gpu import GpuScreen, GpuUnavailable, gate_threshold

OK, WHY = gpumod.available()
needs_gpu = pytest.mark.skipif(not OK, reason=f"GPU 不可用：{WHY}")
FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "mining_cpu_digest.json"


@pytest.fixture(scope="module")
def panel():
    return gc.small_panel()


# ---- CPU 路径不变 -------------------------------------------------------------------


@pytest.mark.parametrize("strategy", sorted(gc.STRATEGIES))
@pytest.mark.parametrize("screen", [None, "cpu"])
def test_cpu_screen_matches_frozen_digest(panel, tmp_path, strategy, screen):
    fixture = json.loads(FIXTURE.read_text())
    here = f"{sys.platform}-{platform.machine()}"
    if here != fixture["platform"]:
        pytest.skip(
            f"冻结摘要在 {fixture['platform']} 上生成，浮点结果只在同一平台逐位可比"
        )
    want = fixture[strategy]
    kw = {} if screen is None else {"screen": screen}
    result = gc.run(panel, tmp_path, strategy, **kw)
    assert gc.mine_digest(result, tmp_path) == want


def test_cpu_screen_does_not_load_gpu(tmp_path):
    here, root = str(Path(__file__).parent), str(tmp_path)
    code = (
        f"import sys; sys.path.insert(0, {here!r})\n"
        "import gpu_cases as gc\n"
        "from pathlib import Path\n"
        f"gc.run(gc.small_panel(), Path({root!r}), 'random', screen='cpu')\n"
        "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules\n"
        "assert 'ashare.factor.mining.gpu' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


def test_screen_argument_validated(panel, tmp_path):
    for bad in ("tpu", "GPU", 1, None):
        with pytest.raises(ValueError, match="screen"):
            gc.run(panel, tmp_path, "random", screen=bad)
    for bad in (
        {"margin": -0.1},
        {"margin": math.nan},
        {"chunk": 0},
        {"memory_limit": 0},
    ):
        with pytest.raises(ValueError):
            GpuScreen(**bad)


def test_gate_threshold():
    """门槛 = min(screen_t, z_{1−α}) − margin；α ≥ 0.5 时全部复算。"""
    assert gate_threshold(1.0, 0.05, 0.3) == pytest.approx(0.7)
    assert gate_threshold(2.0, 0.05, 0.3) == pytest.approx(1.6448536269514722 - 0.3)
    assert gate_threshold(0.5, 0.05, 0.0) == 0.5
    assert gate_threshold(1.0, 0.5, 0.3) == -math.inf


def test_gpu_unavailable_raises(panel, tmp_path, monkeypatch):
    """GPU 不可用时 mine(screen="gpu") 报错，不静默退回 CPU。"""
    from ashare.factor.mining.gpu import backend

    def nope():
        raise GpuUnavailable("测试：没有 GPU")

    monkeypatch.setattr(backend, "probe", nope)
    with pytest.raises(GpuUnavailable, match="没有 GPU"):
        gc.run(panel, tmp_path, "random", screen="gpu")
    assert not (tmp_path / "ledger.jsonl").exists()


# ---- 选择不变性 ---------------------------------------------------------------------

_SUMMARY_SKIP = ("seconds", "per_second", "gpu")


def _rows(result):
    return {r["key"]: r for r in result.candidates}


def _ledger(root):
    from ashare.factor.mining import open_ledger

    out = {}
    for r in open_ledger(None, root).records():
        out[r["key"]] = {k: v for k, v in r.items() if k not in ("time", "code", "gpu")}
    return out


def assert_same_selection(cpu, gpu, root_cpu, root_gpu, alpha=0.05):
    rc, rg = _rows(cpu), _rows(gpu)
    assert list(rc) == list(rg)  # 同一串候选，同样的提出顺序
    sc = {k: v for k, v in cpu.summary.items() if k not in _SUMMARY_SKIP}
    sg = {k: v for k, v in gpu.summary.items() if k not in _SUMMARY_SKIP}
    assert json.dumps(gc._canon(sc), sort_keys=True) == json.dumps(
        gc._canon(sg), sort_keys=True
    )
    assert [r["key"] for r in cpu.selected] == [r["key"] for r in gpu.selected]
    lc, lg = _ledger(root_cpu), _ledger(root_gpu)
    dropped = 0
    for key, a in rc.items():
        b = rg[key]
        assert a["fitness"] == b["fitness"] or (
            a["fitness"] is None and b["fitness"] is None
        )
        if b["stage"] == "gpu_screen":
            dropped += 1
            assert a["stage"] == "screen" and a["status"] == "ok", key
            assert b["train"] is None and b["valid"] is None
            p = a["train"]["ic"]["p"]
            assert p is None or not p <= alpha, key  # 不显著：不改变任何多重检验计数
            continue
        assert gc._canon(a) == gc._canon(b), key
        assert gc._canon(lc[key]) == gc._canon(lg[key]), key
    files_c = gc.view(cpu, root_cpu)["files"]
    files_g = gc.view(gpu, root_gpu)["files"]
    assert files_c == files_g
    return dropped


CASES = [
    ("random", 3, 0.5),
    ("evolution", 3, 0.5),
    ("random", 8, 1.0),
    ("evolution", 8, 2.0),
]


@needs_gpu
@pytest.mark.parametrize(("strategy", "seed", "screen_t"), CASES)
def test_selection_invariance(tmp_path, strategy, seed, screen_t):
    p = gc.small_panel(seed=seed)
    cpu = gc.run(p, tmp_path / "cpu", strategy, screen_t=screen_t, screen="cpu")
    gpu = gc.run(p, tmp_path / "gpu", strategy, screen_t=screen_t, screen="gpu")
    dropped = assert_same_selection(cpu, gpu, tmp_path / "cpu", tmp_path / "gpu")
    g = gpu.summary["gpu"]
    assert (
        dropped == g["dropped"] > 0 and g["rechecked"] > 0
    )  # 两条路都走到了，不是空转
    assert g["screened"] + g["unsupported"] == gpu.summary["evaluated"]


@needs_gpu
def test_gpu_mine_is_deterministic(tmp_path, panel):
    a = gc.run(panel, tmp_path / "a", "evolution", screen="gpu")
    b = gc.run(panel, tmp_path / "b", "evolution", screen="gpu")
    va, vb = gc.view(a, tmp_path / "a"), gc.view(b, tmp_path / "b")
    for v in (va, vb):
        for key in (
            "gpu_seconds",
            "upload_seconds",
            "recheck_seconds",
            "peak_device_memory",
        ):
            v["summary"]["gpu"].pop(key)
    assert json.dumps(va, sort_keys=True) == json.dumps(vb, sort_keys=True)


@needs_gpu
def test_ledger_has_no_float32_numbers_and_is_reused_on_cpu(tmp_path, panel):
    """GPU 判为不过线的记录不带任何指标；之后在同一台账上用 CPU 跑，结果与全新的 CPU 搜索相同
    （复算过的直接复用，GPU 记录按 CPU 重算）。"""
    from ashare.factor.mining import open_ledger

    gc.run(panel, tmp_path / "shared", "random", screen="gpu")
    notes = set()
    for r in open_ledger(None, tmp_path / "shared").records():
        note = r["gpu"]
        notes |= set(note)
        assert note["decision"] in ("drop", "recheck")
        if r["stage"] == "gpu_screen":
            assert r["metrics"] is None and r["screen"] is None
            assert note["decision"] == "drop"
    assert notes <= {"device", "kernel", "margin", "gate_t", "decision", "reason"}
    again = gc.run(panel, tmp_path / "shared", "random", screen="cpu")
    fresh = gc.run(panel, tmp_path / "fresh", "random", screen="cpu")
    assert again.summary["cache_hits"] > 0
    assert [r["key"] for r in again.selected] == [r["key"] for r in fresh.selected]
    assert again.summary["significant"] == fresh.summary["significant"]
    for a, b in zip(again.candidates, fresh.candidates, strict=True):
        a = {k: v for k, v in a.items() if k != "cached"}
        b = {k: v for k, v in b.items() if k != "cached"}
        assert gc._canon(a) == gc._canon(b)


@needs_gpu
def test_non_finite_gpu_t_is_rechecked(tmp_path, panel, monkeypatch):
    """GPU 日 IC 给不出有限 t 值时（全缺失、样本不足）一律送 CPU 复算，宁可多算也不漏：
    让 GPU 对每个候选都只给出全缺失的日 IC，结果必须与纯 CPU 完全相同、一个都不丢。"""
    import numpy as np

    from ashare.factor.mining.gpu import Outcome, Screener

    def all_nan(self, exprs):
        return [Outcome(ic=np.full(self.total_rows, np.nan), noise=0.0) for _ in exprs]

    monkeypatch.setattr(Screener, "daily_ic", all_nan)
    cpu = gc.run(panel, tmp_path / "cpu", "random", screen="cpu")
    gpu = gc.run(panel, tmp_path / "gpu", "random", screen="gpu")
    assert assert_same_selection(cpu, gpu, tmp_path / "cpu", tmp_path / "gpu") == 0
    assert gpu.summary["gpu"]["dropped"] == 0


@needs_gpu
def test_gpu_unable_to_compute_is_rechecked(tmp_path, panel, monkeypatch):
    """来源：2026-11-14 合入前审查的变异检查。GPU 算不了的（算子不支持、超出 float32 范围、
    舍入并列、设备出错，Outcome 只带 reason、没有 ic）必须送 CPU 复算；把它们当成「不过线」
    丢掉时，原有测试全部照过（真实的小空间里没有这类候选）。"""
    from ashare.factor.mining.gpu import Outcome, Screener

    def cannot(self, exprs):
        return [Outcome(reason="unsupported: test") for _ in exprs]

    monkeypatch.setattr(Screener, "daily_ic", cannot)
    cpu = gc.run(panel, tmp_path / "cpu", "random", screen="cpu")
    gpu = gc.run(panel, tmp_path / "gpu", "random", screen="gpu")
    assert assert_same_selection(cpu, gpu, tmp_path / "cpu", tmp_path / "gpu") == 0
    g = gpu.summary["gpu"]
    assert g["dropped"] == 0 and g["rechecked"] == gpu.summary["evaluated"] > 0
