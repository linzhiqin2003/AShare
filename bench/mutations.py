"""逐一修改真实 Rust 源码，重建、证伪、恢复；不靠测试专用开关。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates/panelbt-core/src/lib.rs"
OUT = ROOT / ".cache/mutations"


def command(args, log):
    with log.open("w") as f:
        result = subprocess.run(
            args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=False
        )
    return result.returncode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = CORE.read_text()
    fingerprint = hashlib.sha256(original.encode()).hexdigest()
    assert command(["make", "dev"], OUT / "baseline-build.log") == 0
    assert command(["make", "check"], OUT / "baseline-check.log") == 0, (
        "原源码必须先全部通过"
    )
    mutations = [
        (
            "去掉印花税",
            "amount * rate,",
            "amount * rate * 0.0,",
            "test_fees_and_tax_on_slipped_amount",
        ),
        (
            "忽略 can_sell",
            "!p.is_finite() || !m.sell[base + i]",
            "!p.is_finite()",
            "test_can_sell_and_carry",
        ),
        (
            "现金不足不同比缩减",
            "            cash / total",
            "            1.0",
            "test_proportional_cash_scaling_before_column_order",
        ),
        (
            "最低佣金不生效",
            "(amount * s.commission).max(s.min_commission)",
            "amount * s.commission",
            "test_min_commission_and_cash_scaling",
        ),
        (
            "分红不入账",
            "cash += shares[i] * div;",
            "cash += shares[i] * div * 0.0;",
            "test_dividend_before_split_and_odd_buy",
        ),
        (
            "送转不调股数",
            "shares[i] *= factor;",
            "shares[i] *= 1.0;",
            "test_dividend_before_split_and_odd_buy",
        ),
        (
            "顺延触碰非受限持仓",
            "std::mem::swap(&mut commands, &mut pending);",
            "std::mem::swap(&mut commands, &mut pending);\n            for &i in &held { commands.push((i, weights[i])); }",
            "test_carry_only_blocked_security",
        ),
        (
            "忽略成交量上限",
            "let capacity = cap(m, s, base + i, step);",
            "let capacity = cap(m, s, base + i, step).max(f64::INFINITY);",
            "test_volume_and_odd_lot_liquidation",
        ),
        (
            "卖价滑点方向反转",
            "let price = p * (1.0 - s.slippage);",
            "let price = p * (1.0 + s.slippage);",
            "test_fees_and_tax_on_slipped_amount",
        ),
        (
            "忽略退市回收率",
            "let price = last[i] * s.recovery;",
            "let price = last[i];",
            "test_delist_prior_close_and_no_reentry",
        ),
    ]
    records = []
    try:
        for index, (name, old, new, test) in enumerate(mutations, 1):
            assert original.count(old) == 1, (name, original.count(old))
            CORE.write_text(original.replace(old, new))
            assert command(["make", "dev"], OUT / f"{index:02d}-build.log") == 0, name
            log = OUT / f"{index:02d}-test.log"
            status = command(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "pytest",
                    f"test/test_engine.py::{test}",
                    "-q",
                ],
                log,
            )
            # pytest 1 是断言失败；编译失败、收集错误、未找到测试不算杀死变异。
            assert status == 1 and "1 failed" in log.read_text(), (
                name,
                status,
                log.read_text(),
            )
            records.append({"mutation": name, "test": test, "exit_code": status})
            print(f"{index}/10 已捕获：{name} -> {test}", flush=True)
            CORE.write_text(original)
    finally:
        CORE.write_text(original)
        assert command(["make", "dev"], OUT / "restored-build.log") == 0
    assert hashlib.sha256(CORE.read_bytes()).hexdigest() == fingerprint
    # 新增文件尚未跟踪，使用 --no-index 与原源码比较，避免普通 git diff 漏过新增文件。
    baseline = OUT / "original.rs"
    baseline.write_text(original)
    assert (
        command(
            ["git", "diff", "--no-index", "--exit-code", str(baseline), str(CORE)],
            OUT / "restored-diff.log",
        )
        == 0
    )
    assert command(["make", "check"], OUT / "restored-check.log") == 0
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "source_sha256": fingerprint,
                "mutations": records,
                "restored_git_diff_exit": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        "10/10 变异捕获；源码哈希一致，git diff 无差异，恢复后 make check 全绿。",
        flush=True,
    )


if __name__ == "__main__":
    main()
