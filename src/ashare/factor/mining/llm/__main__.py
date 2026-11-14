"""命令行：用大模型研究循环跑一次 `mine`，打印结果与费用。

    python -m ashare.factor.mining.llm --rounds 20 --batch 8 --budget-usd 2
    python -m ashare.factor.mining.llm --synthetic --rounds 5      # 合成行情，试流程

真实数据用 ``ashare.data.open()`` 的面板、`Split.standard()`（训练 2016 起、验证 2025 年、
留出 2026 年起）；合成行情按行数切（前 60% 训练、再 20% 验证、最后 20% 留出）。台账、指纹和
转写记录写到 ``--root``（默认数据根目录下不入库的 ``mining/``）。密钥读 ``DEEPSEEK_API_KEY``
（``--key-env`` 可改）。
"""

from __future__ import annotations

import argparse
import json
import sys


def parser():
    ap = argparse.ArgumentParser(prog="python -m ashare.factor.mining.llm")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--budget-tokens", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--key-env", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="每次回答的输出上限（含思考）")  # fmt: skip
    ap.add_argument("--reasoning", choices=("off", "low", "medium", "high"), default=None,
                    help="思考强度；不传用模型默认。只支持 OpenRouter")  # fmt: skip
    ap.add_argument(
        "--price-in",
        type=float,
        default=None,
        help="每百万输入 token（未命中缓存）美元",
    )
    ap.add_argument(
        "--price-cached",
        type=float,
        default=None,
        help="每百万命中缓存的输入 token 美元",
    )
    ap.add_argument(
        "--price-out", type=float, default=None, help="每百万输出 token 美元"
    )
    ap.add_argument(
        "--root", default=None, help="台账与转写记录目录，默认数据根目录下的 mining/"
    )
    ap.add_argument("--screen", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--screen-t", type=float, default=1.0)
    ap.add_argument("--start", default="20160101", help="真实数据面板的起始日")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--synthetic", action="store_true", help="用合成行情（埋了一个已知信号）"
    )
    ap.add_argument("--days", type=int, default=900, help="合成行情的天数")
    ap.add_argument("--stocks", type=int, default=150, help="合成行情的股票数")
    ap.add_argument("--top", type=int, default=10, help="打印的候选个数")
    return ap


def _client(args):
    from dataclasses import replace

    from .client import (
        DEFAULT_BASE_URL,
        DEFAULT_KEY_ENV,
        DEFAULT_MODEL,
        ChatClient,
        Pricing,
    )

    pricing = Pricing()
    for name, attr in (("price_in", "input"), ("price_cached", "cached_input"),
                       ("price_out", "output")):  # fmt: skip
        if getattr(args, name) is not None:
            pricing = replace(pricing, **{attr: getattr(args, name)})
    return ChatClient(
        args.model or DEFAULT_MODEL,
        base_url=args.base_url or DEFAULT_BASE_URL,
        api_key_env=args.key_env or DEFAULT_KEY_ENV,
        pricing=pricing,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
    )


def _data(args, space):
    from ..search import Split

    if args.synthetic:
        from ..synthetic import market

        panel = market(t=args.days, n=args.stocks, seed=args.seed, strength=0.003)
        d, t = panel.dates, len(panel.dates)
        return (
            panel,
            Split(valid_start=str(d[int(t * 0.6)]), holdout_start=str(d[int(t * 0.8)])),
            {},
        )
    import ashare.data as ad
    from ashare.backtest import market_from_panel
    from ashare.factor.evaluate import FactorEvaluator

    data = ad.open()
    fields = sorted(set(FactorEvaluator.required_fields()) | set(space.fields))
    panel = data.panel(fields, start=args.start)
    market = market_from_panel(data, panel=panel, validate=False)
    return panel, Split.standard(), {"market": market}


def main(argv=None, client=None, out=None):
    from ..search import mine
    from ..space import Space
    from .strategy import LLMResearch

    out = out or sys.stdout
    args = parser().parse_args(argv)
    space = Space.default()
    panel, split, extra = _data(args, space)
    if args.threads is not None:
        extra["threads"] = args.threads
    strategy = LLMResearch(
        client or _client(args),
        rounds=args.rounds,
        batch=args.batch,
        seed=args.seed,
        budget_usd=args.budget_usd,
        budget_tokens=args.budget_tokens,
        patience=args.patience,
    )
    result = mine(panel, split=split, space=space, strategy=strategy, root=args.root,
                  screen=args.screen, screen_t=args.screen_t, **extra)  # fmt: skip
    run = strategy.last_run or {}
    print(result.text(top=args.top), file=out)
    print(
        f"\n大模型：{run.get('rounds', 0)} 轮，停止原因 {run.get('stop')}，"
        f"输入 {run.get('prompt_tokens', 0)} / 输出 {run.get('completion_tokens', 0)} token，"
        f"费用 ${run.get('cost_usd', 0.0):.4f}；评估 {run.get('evaluated', 0)} 个、"
        f"不合法 {run.get('invalid', 0)} 个、重复 {run.get('duplicates', 0)} 个",
        file=out,
    )
    if run.get("error"):
        label = "接口错误" if run.get("stop") == "llm_error" else "停止说明"
        print(f"{label}：{run['error']}", file=out)
    print(f"转写记录：{run.get('transcript')}", file=out)
    return {"summary": json.loads(json.dumps(result.summary, default=str)), "llm": run}


if __name__ == "__main__":
    main()
