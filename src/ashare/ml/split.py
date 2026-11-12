"""按时间切分：滚动窗口或扩展窗口，训练 → 间隔 → 验证 → 间隔 → 预测。

间隔（purge）由标签定义推出：信号日 s 的标签在第 s + lag + h 行才结束，所以两段之间
至少跳过 `span = lag + h` 个交易日，前一段最后一个样本的标签结束日才严格早于后一段的
第一天。`gap` 可以加大（相当于 embargo），不能小于 span。

每一折的区间都是左闭右开的行号 `[a, b)`；行号就是交易日。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

RETRAIN = ("month", "quarter", "halfyear", "year")


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[int, int]
    valid: tuple[int, int] | None
    test: tuple[int, int]


@dataclass(frozen=True)
class Schedule:
    folds: tuple[Fold, ...]
    dates: np.ndarray
    span: int
    gap: int
    settings: dict

    def __len__(self):
        return len(self.folds)

    def __iter__(self):
        return iter(self.folds)

    @property
    def digest(self):
        text = json.dumps(
            [
                "schedule",
                1,
                self.span,
                self.gap,
                [asdict(f) for f in self.folds],
                str(self.dates[0]) if len(self.dates) else "",
                len(self.dates),
            ],
            sort_keys=True,
        )
        return hashlib.sha256(text.encode()).hexdigest()

    def table(self):
        """每折一行的字典列表（日期为 YYYYMMDD）。"""
        d = self.dates
        out = []
        for f in self.folds:
            row = {
                "fold": f.index,
                "train": (str(d[f.train[0]]), str(d[f.train[1] - 1])),
                "train_days": f.train[1] - f.train[0],
                "valid": None
                if f.valid is None
                else (str(d[f.valid[0]]), str(d[f.valid[1] - 1])),
                "valid_days": 0 if f.valid is None else f.valid[1] - f.valid[0],
                "test": (str(d[f.test[0]]), str(d[f.test[1] - 1])),
                "test_days": f.test[1] - f.test[0],
            }
            out.append(row)
        return out

    def describe(self):
        """切分日程的文本表格，方便人工检查间隔。"""
        lines = [
            f"切分：{self.settings.get('window')} 窗口，重训 {self.settings.get('retrain')}，"
            f"间隔 {self.gap} 个交易日（标签跨度 {self.span}），共 {len(self.folds)} 折"
        ]
        lines.append(
            f"{'折':>3}  {'训练':<19} {'天':>5}  {'验证':<19} {'天':>4}  {'预测':<19} {'天':>4}"
        )
        for r in self.table():
            valid = "—" if r["valid"] is None else f"{r['valid'][0]}–{r['valid'][1]}"
            lines.append(
                f"{r['fold']:>3}  {r['train'][0]}–{r['train'][1]} {r['train_days']:>5}  "
                f"{valid:<19} {r['valid_days']:>4}  {r['test'][0]}–{r['test'][1]} "
                f"{r['test_days']:>4}"
            )
        for note in self.settings.get("skipped", []):
            lines.append(f"跳过：{note}")
        return "\n".join(lines)

    def __str__(self):
        return self.describe()


def _period_keys(dates, retrain):
    d = np.asarray(dates).astype(str)
    year = np.array([int(x[:4]) for x in d])
    month = np.array([int(x[4:6]) for x in d])
    if retrain == "month":
        return year * 12 + month
    if retrain == "quarter":
        return year * 4 + (month - 1) // 3
    if retrain == "halfyear":
        return year * 2 + (month - 1) // 6
    return year


def _row(dates, value, side="left"):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    text = str(value).replace("-", "")
    return int(np.searchsorted(np.asarray(dates).astype(str), text, side=side))


def _years_back(dates, row, years):
    d = str(np.asarray(dates)[row])
    y = int(d[:4]) - int(years)
    target = f"{y:04d}{d[4:]}"
    return int(np.searchsorted(np.asarray(dates).astype(str), target, side="left"))


def schedule(
    dates,
    span,
    *,
    start=None,
    end=None,
    retrain="quarter",
    window="rolling",
    train_years=5,
    train_days=None,
    valid_days=120,
    gap=None,
    min_train_days=None,
    first=0,
):
    """生成切分日程。

    dates：面板日期（行 = 交易日）；span：标签跨度 lag + h（`Label.span`）。
    start / end：预测期的首末日（日期字符串或行号；end 含当天，默认到末尾）。
    retrain：`"month"`、`"quarter"`、`"halfyear"`、`"year"` 或正整数 k（每 k 个交易日）。
    window：`"rolling"`（训练期取 train_years 年或 train_days 天）或 `"expanding"`（从
    first 行开始）。valid_days：训练期末尾切出的验证天数（0 表示不要验证集）。
    gap：每两段之间跳过的交易日数，默认等于 span，不能更小。
    min_train_days：训练天数不足的折跳过（默认 train_days，没给时 250）（这些日子不出预测），并在 describe 里列出。
    """
    dates = np.asarray(dates)
    t = len(dates)
    if isinstance(span, bool) or not isinstance(span, (int, np.integer)) or span < 1:
        raise ValueError("span: 需要正整数（标签的 lag + horizon）")
    span = int(span)
    if gap is None:
        gap = span
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < span:
        raise ValueError(
            f"gap: 不能小于标签跨度 {span}（lag + horizon），否则训练样本的标签会和后一段重叠"
        )
    gap = int(gap)
    if window not in ("rolling", "expanding"):
        raise ValueError("window: 只能是 rolling 或 expanding")
    if valid_days is None or valid_days < 0:
        raise ValueError("valid_days: 需要非负整数")
    if min_train_days is None:
        min_train_days = (
            int(train_days) if train_days is not None and window == "rolling" else 250
        )
    lo = _row(dates, start) if start is not None else 0
    hi = _row(dates, end, side="right") if end is not None else t
    lo, hi = max(lo, 0), min(hi, t)
    if lo >= hi:
        raise ValueError("start 必须早于 end")
    if isinstance(retrain, (int, np.integer)) and not isinstance(retrain, bool):
        if retrain < 1:
            raise ValueError("retrain: 正整数")
        points = list(range(lo, hi, int(retrain)))
    elif retrain in RETRAIN:
        keys = _period_keys(dates, retrain)
        change = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        points = [lo] + [int(p) for p in change if lo < p < hi]
    else:
        raise ValueError(f"retrain: 只能是 {RETRAIN} 或正整数")
    points.append(hi)
    folds, skipped = [], []
    for k in range(len(points) - 1):
        test = (points[k], points[k + 1])
        if valid_days:
            v_end = test[0] - gap
            valid = (v_end - valid_days, v_end)
            train_end = valid[0] - gap
        else:
            valid = None
            train_end = test[0] - gap
        if window == "expanding":
            train_start = first
        elif train_days is not None:
            train_start = train_end - int(train_days)
        else:
            train_start = _years_back(dates, max(train_end - 1, 0), train_years)
        train_start = max(train_start, first, 0)
        if train_end - train_start < min_train_days or (valid and valid[0] < first):
            skipped.append(
                f"{dates[test[0]]}–{dates[test[1] - 1]}：训练天数 "
                f"{max(train_end - train_start, 0)} < {min_train_days}"
            )
            continue
        folds.append(Fold(len(folds), (train_start, train_end), valid, test))
    settings = {
        "retrain": retrain,
        "window": window,
        "train_years": train_years,
        "train_days": train_days,
        "valid_days": valid_days,
        "gap": gap,
        "min_train_days": min_train_days,
        "skipped": skipped,
    }
    sched = Schedule(tuple(folds), dates, span, gap, settings)
    check_schedule(sched)
    return sched


def check_schedule(sched, span=None):
    """确认每折：训练、验证、预测三段按时间先后，且任一训练/验证样本的标签结束日
    （信号日 + span）严格早于下一段的第一天。不满足就抛 ValueError。"""
    span = sched.span if span is None else span
    for f in sched.folds:
        segments = [f.train] + ([f.valid] if f.valid else []) + [f.test]
        for a, b in segments:
            if not 0 <= a < b <= len(sched.dates):
                raise ValueError(f"第 {f.index} 折区间无效：{(a, b)}")
        for (_a0, b0), (a1, _) in zip(segments, segments[1:], strict=False):
            if (b0 - 1) + span >= a1:
                raise ValueError(
                    f"第 {f.index} 折：前一段最后一天 {b0 - 1} 的标签结束日 {b0 - 1 + span} "
                    f"不早于后一段第一天 {a1}"
                )
    return True
