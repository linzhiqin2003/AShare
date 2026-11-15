"""同一小时边界上的现货与 U 本位钱包；策略决策先冻结，再同步入账。"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import panelbt as bt

from ashare.data.crypto.events import asof
from ashare.data.crypto.timeutil import DAY, HOUR

from .crypto_run import InvalidCryptoResult
from .crypto_schedule import DailyTargets, hour


@dataclass(frozen=True)
class MarginBand:
    """权益/上一结算名义越界时补到 target；正数表示现货划入合约。"""

    target: float
    low: float
    high: float
    pre_fund: bool = True
    buffer: float = 0.02

    def __post_init__(self):
        if not np.isfinite(self.buffer) or self.buffer < 0:
            raise ValueError("MarginBand buffer 必须为非负有限值")
        if not all(np.isfinite(x) for x in (self.target, self.low, self.high)) or not (
            0 <= self.low < self.target < self.high
        ):
            raise ValueError("MarginBand 需要 0 <= low < target < high")

    def __call__(self, t, spot_state, usdm_state):
        nominal = usdm_state.notional
        if nominal == 0:
            return 0.0
        equity = usdm_state.wallet_balance + usdm_state.unrealized
        ratio = equity / nominal
        return (
            self.target * nominal - equity
            if ratio < self.low or ratio > self.high
            else 0.0
        )

    def config(self):
        return asdict(self)

    def state(self):
        return asdict(self)

    def restore(self, state):
        if state != asdict(self):
            raise ValueError("MarginBand 参数不一致")


def _contracts(value):
    """币数换张数：仅在距离整数不超过 1e-9 张时吸附，其他值向零取整。"""
    nearest = np.rint(value)
    return np.trunc(np.where(np.abs(value - nearest) <= 1e-9, nearest, value))


def _frozen(state):
    values = {}
    for f in fields(state):
        v = getattr(state, f.name)
        if isinstance(v, np.ndarray):
            v = v.copy()
            v.flags.writeable = False
        values[f.name] = v
    return type(state)(**values)


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _strategy(strategy, *, require=False):
    if strategy is None:
        return None
    protocol = callable(getattr(strategy, "state", None)) and callable(
        getattr(strategy, "restore", None)
    )
    if require and not protocol:
        raise ValueError("自定义策略快照需要 state() / restore(state)")
    config = getattr(strategy, "config", None)
    if require and not callable(config):
        raise ValueError(
            "自定义策略快照需要 config() 返回全部固定参数，state() / restore() 管理可变状态"
        )
    cls = type(strategy)
    name = getattr(strategy, "__qualname__", cls.__qualname__)
    module = getattr(strategy, "__module__", cls.__module__)
    return {
        "class": module + "." + name,
        "config": copy.deepcopy(config()) if callable(config) else None,
    }


class _IsolatedStrategy:
    """隔离 deepcopy 仍返回自身的函数闭包，通过协议交换私有状态。"""

    def __init__(self, source):
        _strategy(source, require=True)
        self._source = source
        self._state = copy.deepcopy(source.state())
        self.__module__ = getattr(source, "__module__", type(source).__module__)
        self.__qualname__ = getattr(source, "__qualname__", type(source).__qualname__)

    def _invoke(self, state, args=None):
        saved = copy.deepcopy(self._source.state())
        try:
            self._source.restore(copy.deepcopy(state))
            result = None if args is None else self._source(*args)
            updated = copy.deepcopy(self._source.state())
            return result, updated
        finally:
            self._source.restore(saved)

    def __call__(self, *args):
        result, self._state = self._invoke(self._state, args)
        return result

    def config(self):
        return copy.deepcopy(self._source.config())

    def state(self):
        return copy.deepcopy(self._state)

    def restore(self, state):
        _, self._state = self._invoke(state)


def _clone_strategy(value):
    cloned = copy.deepcopy(value)
    if callable(value) and cloned is value:
        return _IsolatedStrategy(value)
    return cloned


@dataclass
class TwoWalletResult:
    spot: bt.Result
    usdm: bt.Result
    times: np.ndarray
    coins: tuple[str, ...]
    equity: np.ndarray
    nav: np.ndarray
    returns: np.ndarray
    residual_coins: np.ndarray
    residual_notional: np.ndarray
    rounding_residual_notional: np.ndarray
    underfill_residual_notional: np.ndarray
    transfers: list[dict]
    diagnostics: dict

    def daily(self, trade_hour=0):
        """完整 24 小时区间的末值；报表年化因子 365。"""
        import polars as pl

        labels = (self.times - hour(trade_hour)) // DAY * DAY + hour(trade_hour)
        return (
            pl.DataFrame(dict(day=labels, nav=self.nav, equity=self.equity))
            .group_by("day", maintain_order=True)
            .agg(pl.col("nav").last(), pl.col("equity").last(), pl.len().alias("hours"))
            .filter(pl.col("hours") == 24)
        )


class TwoWallet:
    """逐小时两钱包。目标为币→(现货,合约)，或 C×2 数组。

    kind='coins' 是基础币数，kind='weights' 是合计权益的权重。
    targets 可为 T×C×2、两个 DailyTargets，或 (t, spot_state, usdm_state)
    回调。直接 step(target) 不移位；日线必须先经 daily_targets 唯一一次移位。
    """

    def __init__(
        self,
        spot,
        usdm,
        targets=None,
        *,
        transfer=None,
        hedge="independent",
        trade_hour=None,
        initial=None,
        kind="coins",
        residual_limit=1e-8,
        allow_residual=False,
        allow_tier_exceeded=False,
        transfer_margin_factor=1.0,
        record_trades=None,
    ):
        if spot.kind != "spot" or usdm.kind != "usdm":
            raise ValueError("需要现货钱包与 U 本位合约钱包")
        if not np.array_equal(spot.times, usdm.times) or np.any(
            np.diff(spot.times) != HOUR
        ):
            raise ValueError("两个钱包必须为同一连续小时时间轴")
        if not len(spot.times) or np.any(spot.times % HOUR):
            raise ValueError("时间轴必须非空且位于 UTC 整点")
        if usdm.futures is None or usdm.futures.liquidation is None:
            raise ValueError("U 本位钱包需要交易所强平与 W/U 账本")
        if hedge not in ("follow_spot", "independent") or kind not in (
            "coins",
            "weights",
        ):
            raise ValueError("hedge/kind 参数不合法")
        if not np.isfinite(residual_limit) or residual_limit < 0:
            raise ValueError("residual_limit 必须是非负有限 USDT 名义")
        if not np.isfinite(transfer_margin_factor) or transfer_margin_factor < 1:
            raise ValueError("transfer_margin_factor 必须 >= 1；只支持收紧限额的敏感性")
        if (
            isinstance(targets, tuple)
            and len(targets) == 2
            and all(isinstance(x, DailyTargets) for x in targets)
        ):
            if targets[0].trade_hour != targets[1].trade_hour or (
                trade_hour is not None and trade_hour != targets[0].trade_hour
            ):
                raise ValueError("两腿日频 trade_hour 不一致")
            trade_hour = targets[0].trade_hour
        if trade_hour is not None:
            hour(trade_hour)
        self.spot_market, self.usdm_market = spot, usdm
        self.times = spot.times.copy()
        self.coins = tuple(
            dict.fromkeys(p["coin"] for m in (spot, usdm) for p in m.profiles)
        )
        self._indices = []
        self._units = []
        for m in (spot, usdm):
            names = [p["coin"] for p in m.profiles]
            if len(set(names)) != len(names):
                raise ValueError(
                    "每个钱包每币只允许一列；多季度/永续选择须由调用方明确"
                )
            self._indices.append(np.array([self.coins.index(c) for c in names]))
            units = np.asarray(
                m.arrays.get("coin_units", [[p["coin_units"] for p in m.profiles]]),
                float,
            )
            units = np.broadcast_to(units, m.market.shape).copy()
            if not np.isfinite(units).all() or np.any(units <= 0):
                raise ValueError("coin_units 必须为有限正数")
            self._units.append(units)
        self._mult = np.asarray(usdm.futures.multiplier)
        self.hedge, self.kind, self.trade_hour = hedge, kind, trade_hour
        self.residual_limit, self.allow_residual = residual_limit, allow_residual
        self.allow_tier_exceeded = allow_tier_exceeded
        self.transfer_margin_factor = transfer_margin_factor
        self.transfer = transfer
        self._strategy_initial = _strategy(transfer)
        self.targets = self._prepare_targets(targets)
        self._target_initial = _strategy(targets) if callable(targets) else None
        initial = initial or (spot.config.initial_cash, usdm.config.initial_cash)
        if len(initial) != 2 or not np.isfinite(initial).all() or min(initial) <= 0:
            raise ValueError("initial 需要两个正的初始 USDT 余额")
        self.initial = tuple(float(x) for x in initial)
        configs = []
        for m, amount in zip((spot, usdm), self.initial, strict=True):
            if np.any(np.asarray(m.config.cash_flow) != 0):
                raise ValueError("两钱包不允许静态 cash_flow；内部划转由协调器控制")
            if (
                m.config.margin.allow_short
                or m.config.margin.allow_financing
                or m.config.rules is not None
            ):
                raise ValueError("两钱包不允许现货借币/融资或通用退出规则")
            configs.append(
                replace(
                    m.config,
                    initial_cash=amount,
                    carry_blocked=False
                    if trade_hour is not None
                    else m.config.carry_blocked,
                    record_trades=m.config.record_trades
                    if record_trades is None
                    else record_trades,
                )
            )
        self._configs = configs
        self._s = bt.Simulator(
            replace(spot.market, crypto_spot=True),
            spot.costs,
            configs[0],
            kind="shares",
            fingerprint="full",
        )
        self._u = bt.Simulator(
            usdm.market, usdm.costs, configs[1], kind="shares", fingerprint="full"
        )
        self._spot_shares = self._s.state.shares
        self._usdm_shares = self._u.state.shares
        self._day = 0
        self._invalid = False
        self._busy = False
        self._ratio = np.ones(len(self.coins))
        self._exit_pending = np.zeros(len(self.coins), bool)
        self._equity = np.zeros(len(self.times))
        self._residual = np.zeros((len(self.times), len(self.coins)))
        self._residual_value = np.zeros_like(self._residual)
        self._rounding_value = np.zeros_like(self._residual)
        self._underfill_value = np.zeros_like(self._residual)
        self.transfers = []
        self._has_exits = any(e["kind"] == "delist" for e in spot.events)
        self._fingerprint = self._digest()

    def _prepare_targets(self, targets):
        if targets is None or callable(targets):
            return targets
        if (
            isinstance(targets, tuple)
            and len(targets) == 2
            and all(isinstance(x, DailyTargets) for x in targets)
        ):
            targets = np.stack(
                [x.align(self.times, self.coins) for x in targets], axis=-1
            )
        a = np.asarray(targets, float)
        if a.shape != (len(self.times), len(self.coins), 2) or np.isinf(a).any():
            raise ValueError("targets 需要 T×币数×2，不得含 infinity")
        return a.copy()

    def _digest(self):
        h = hashlib.sha256()
        config = {
            k: getattr(self, k)
            for k in (
                "coins",
                "initial",
                "kind",
                "hedge",
                "trade_hour",
                "residual_limit",
                "allow_residual",
                "allow_tier_exceeded",
                "transfer_margin_factor",
            )
        }
        config.update(
            transfer=self._strategy_initial,
            targets=self._target_initial,
            symbols=[self.spot_market.symbols, self.usdm_market.symbols],
        )
        h.update(_json(config))
        for a in (self.times, *self._indices, *self._units, self._mult):
            h.update(f"{a.shape}{a.dtype}".encode())
            h.update(a.tobytes())
        if isinstance(self.targets, np.ndarray):
            h.update(self.targets.tobytes())
        # 公告版本与信号价格会影响退出和目标换算，也必须纳入指纹。
        for m in (self.spot_market, self.usdm_market):
            h.update(_json(m.events))
            h.update(np.asarray(m.signal_prices).tobytes())
            for name in ("execution_symbol", "lifecycle"):
                h.update(_json(np.asarray(m.arrays[name]).tolist()))
        return h.hexdigest()

    def _check(self):
        if self._invalid:
            raise RuntimeError("两钱包协调器已失效；不能读取半行状态、续跑或快照")

    @property
    def day(self):
        self._check()
        return self._day

    @property
    def state(self):
        self._check()
        if self._busy:
            raise RuntimeError("两钱包尚未完成同一行")
        return _frozen(self._s.state), _frozen(self._u.state)

    def _reference(self, t):
        ref = np.full(len(self.coins), np.nan)
        if t == 0:
            return ref
        for side, m in enumerate((self.usdm_market, self.spot_market)):
            k = 1 - side
            for j, coin in enumerate(self._indices[k]):
                if k == 0:
                    ref[coin] = m.signal_prices[t, j] / self._units[k][t - 1, j]
                elif np.isfinite(m.signal_prices[t, j]):
                    ref[coin] = (
                        m.futures.settlement[t - 1, j] / self._units[k][t - 1, j]
                    )
        return ref

    def _target(self, target, ref, equity):
        if target is None:
            return None
        if isinstance(target, dict):
            unknown = set(target) - set(self.coins)
            if unknown:
                raise ValueError(f"未知币目标：{sorted(unknown)}")
            target = [target.get(c, (0, 0)) for c in self.coins]
        a = np.asarray(target, float)
        if a.shape != (len(self.coins), 2) or np.isinf(a).any():
            raise ValueError("单行目标需要币数×2 的有限值（全 NaN 表示无目标）")
        if np.isnan(a).all():
            return None
        a = np.nan_to_num(a).copy()
        if np.any(a[:, 0] < 0):
            raise ValueError("现货腿只做多")
        if np.any(np.any(a != 0, axis=1) & (~np.isfinite(ref) | (ref <= 0))):
            raise ValueError(
                "非零目标需要上一根已完成的参考价；首行/新上市不得回退本行价格"
            )
        for side in (0, 1):
            absent = np.ones(len(self.coins), bool)
            absent[self._indices[side]] = False
            if np.any(a[absent, side] != 0):
                raise ValueError("目标币缺少对应腿的市场列")
        if self.hedge == "follow_spot":
            if np.any(a[:, 1] > 0) or np.any((a[:, 0] == 0) & (a[:, 1] != 0)):
                raise ValueError("follow_spot 要求现货正目标与非正合约目标成对")
        if self.kind == "weights":
            a *= np.divide(
                equity, ref, out=np.zeros_like(ref), where=np.isfinite(ref) & (ref > 0)
            )[:, None]
        return a

    def _exit_target(self, t, target, state):
        if not self._has_exits and not self._exit_pending.any():
            return target, np.zeros(len(self.coins), bool)
        exits = np.zeros(len(self.spot_market.symbols), bool)
        for e in asof(self.spot_market.events, int(self.times[t])):
            if e["kind"] == "delist":
                exits |= (
                    self.spot_market.arrays["execution_symbol"][t] == e["instrument"]
                )
        self._exit_pending[self._indices[0][exits]] = True
        affected = self._exit_pending.copy()
        if not affected.any():
            return target, np.zeros(len(self.coins), bool)
        exits = affected[self._indices[0]]
        if target is None:
            target = state.shares.copy()
            if self._configs[0].carry_blocked:
                target[state.pending_cols] = state.pending_targets
        target[exits] = 0.0
        return target, affected

    def _prefund(self, t, starget, utarget, ratio, state, requested, available=np.inf):
        # 决策仍只读取上一结算价；手续费和当前生效的规则属于已知参数。
        f = self.usdm_market.futures
        prices = f.settlement[max(t - 1, 0)]
        quantities = utarget.copy()
        if self.hedge == "follow_spot":
            coins = np.zeros(len(self.coins))
            coins[self._indices[0]] = starget * self._units[0][t]
            quantities = _contracts(
                -ratio[self._indices[1]]
                * coins[self._indices[1]]
                / (self._mult * self._units[1][t])
            )
        needed = (quantities != 0) | (state.shares != 0)
        if np.any(needed & (~np.isfinite(prices) | (prices <= 0))):
            raise ValueError("调仓前补资需要有效的上一结算价")
        prices = np.where(needed, prices, 0.0)
        nominal = np.abs(quantities) * self._mult * prices

        def row(value):
            return np.broadcast_to(value, self.usdm_market.market.shape)[t]

        im = float(np.sum(nominal * row(f.margin_rate)))
        turnover = np.abs(quantities - state.shares)
        fees = float(
            np.sum(
                turnover
                * (self._mult * prices * row(f.fee_rate) + row(f.fee_per_contract))
            )
        )
        # 保留原仓开盘所需 IM，同时为新目标的费用及价格变动留名义缓冲。
        required = (
            max(im, state.initial_margin)
            + fees
            + float(nominal.sum()) * self.transfer.buffer
        )
        requested = max(
            requested,
            required - state.wallet_balance - state.unrealized,
            required - state.cash,
        )
        if state.bankrupt:
            return requested
        # 以同一 Rust 成交判定核验扣费后的篮子及逐单保证金，不在 Python 复制比较式。
        while not self._u.opening_feasible(
            quantities, prices, flow=min(requested, available)
        ):
            if requested >= available:
                # 达到硬上限仍不够；保留未满足请求，使日志标记 capacity。
                return max(requested, float(np.nextafter(available, np.inf)))
            # 请求为零/极小时，nextafter(requested) 可能永远不改变入账余额。
            # 步进到下一个可表示的入账余额；这些中间请求的判定完全相同。
            requested = min(
                available,
                max(
                    float(np.nextafter(requested, np.inf)),
                    float(np.nextafter(state.cash + requested, np.inf)) - state.cash,
                ),
            )
        return requested

    def step(self, target=None):
        self._check()
        if self._busy or self._day >= len(self.times):
            raise RuntimeError("不能重复推进；模拟已完成或正在推进")
        t = self._day
        s, u = self.state
        scheduled = self.trade_hour is None or self.times[t] % DAY == hour(
            self.trade_hour
        )
        if not scheduled:
            target = None
        elif target is None and self.targets is not None:
            target = (
                self.targets(t, s, u) if callable(self.targets) else self.targets[t]
            )
        ref = self._reference(t)
        decision = self._target(target, ref, s.equity + u.wallet_balance + u.unrealized)
        ratio = self._ratio.copy()
        if decision is not None and self.hedge == "follow_spot":
            selected = decision[:, 0] > 0
            ratio[selected] = -decision[selected, 1] / decision[selected, 0]
        starget = (
            None
            if decision is None
            else decision[self._indices[0], 0] / self._units[0][t]
        )
        utarget = (
            None
            if decision is None
            else _contracts(
                decision[self._indices[1], 1] / (self._units[1][t] * self._mult)
            )
        )
        starget, exit_override = self._exit_target(t, starget, s)
        requested = 0.0 if self.transfer is None else float(self.transfer(t, s, u))
        if (
            isinstance(self.transfer, MarginBand)
            and self.transfer.pre_fund
            and decision is not None
        ):
            requested = self._prefund(
                t, starget, utarget, ratio, u, requested, s.transferable
            )
        if not np.isfinite(requested):
            raise ValueError("划转策略必须返回有限 USDT 金额")
        available = (
            s.transferable
            if requested >= 0
            else max(
                0.0,
                u.wallet_balance
                + min(u.unrealized, 0.0)
                - self.transfer_margin_factor * u.initial_margin,
            )
        )
        if requested < 0:
            available = min(available, u.transferable)
        amount = min(abs(requested), available)
        reason = "capacity" if amount < abs(requested) else "ok"
        if requested > 0 and u.bankrupt:
            amount, reason = 0.0, "receiver_bankrupt"
        elif requested < 0 and u.bankrupt:
            amount, reason = 0.0, "sender_bankrupt"
        accepted = amount if requested >= 0 else -amount
        self._busy = True
        try:
            sr = self._s.step(starget, flow=-accepted)
            if self.hedge == "follow_spot" and (
                decision is not None
                or exit_override.any()
                or (
                    self.trade_hour is None
                    and not np.array_equal(self._spot_shares, s.shares)
                )
            ):
                actual = np.zeros(len(self.coins))
                actual[self._indices[0]] = self._spot_shares * self._units[0][t]
                following = _contracts(
                    -ratio[self._indices[1]]
                    * actual[self._indices[1]]
                    / (self._units[1][t] * self._mult)
                )
                if decision is None and self.trade_hour is not None:
                    utarget = np.where(
                        exit_override[self._indices[1]], following, u.shares
                    )
                else:
                    utarget = following
            ur = self._u.step(utarget, flow=accepted)
            if sr.cash_flow != -accepted or ur.cash_flow != accepted:
                raise RuntimeError("内核实际划转与预先接受金额不一致")
            if exit_override.any():
                # 两腿都完成退出才释放意图；日频禁用 carry_blocked 不影响此重试。
                remaining = np.zeros(len(self.coins), bool)
                remaining[self._indices[0]] |= self._spot_shares != 0
                remaining[self._indices[1]] |= self._usdm_shares != 0
                for side, sim in enumerate((self._s, self._u)):
                    remaining[self._indices[side][sim.state.pending_cols]] = True
                self._exit_pending &= remaining
            residual = np.zeros(len(self.coins))
            residual[self._indices[0]] += self._spot_shares * self._units[0][t]
            residual[self._indices[1]] += (
                self._usdm_shares * self._mult * self._units[1][t]
            )
            valuation = ref
            if np.any((residual != 0) & ~np.isfinite(ref)):
                # 下架残仓仍按上一已知持仓估值计暴露；禁止将此回退用于新目标。
                valuation = ref.copy()
                for j, coin in enumerate(self._indices[0]):
                    if not np.isfinite(valuation[coin]) and s.shares[j] != 0:
                        valuation[coin] = s.last[j] / self._units[0][t, j]
            value = np.where(residual == 0, 0.0, residual * valuation)
            if self.hedge == "follow_spot":
                desired_coins = np.zeros(len(self.coins))
                desired_coins[self._indices[0]] = (
                    -ratio[self._indices[0]] * self._spot_shares * self._units[0][t]
                )
                desired = _contracts(
                    desired_coins[self._indices[1]] / (self._mult * self._units[1][t])
                )
                f = self.usdm_market.futures
                steps = np.ones_like(desired) if f.qty_step is None else f.qty_step[t]
                quantized = (
                    np.sign(desired) * np.floor(np.abs(desired) / steps + 1e-9) * steps
                )
                minimum = np.ones_like(desired) if f.min_qty is None else f.min_qty[t]
                quantized = np.where(np.abs(quantized) < minimum, 0, quantized)
                quantized_coins = quantized * self._mult * self._units[1][t]
                idx = self._indices[1]
                self._rounding_value[t, idx] = np.nan_to_num(
                    (quantized_coins - desired_coins[idx]) * valuation[idx]
                )
                self._underfill_value[t, idx] = np.nan_to_num(
                    (
                        self._usdm_shares * self._mult * self._units[1][t]
                        - quantized_coins
                    )
                    * valuation[idx]
                )
            self._equity[t] = sr.equity + ur.wallet_balance + ur.unrealized
            self._residual[t], self._residual_value[t] = residual, value
            self._ratio = ratio
            if requested:
                self.transfers.append(
                    dict(
                        row=t,
                        time=int(self.times[t]),
                        direction="spot_to_usdm" if requested > 0 else "usdm_to_spot",
                        requested=abs(requested),
                        actual=amount,
                        reason=reason,
                        rescue_failed=requested > 0 and amount < requested,
                    )
                )
            self._day += 1
        except Exception:
            self._invalid = True
            raise
        finally:
            self._busy = False
        return sr, ur

    def run(self):
        while self.day < len(self.times):
            self.step()
        return self.result()

    def result(self):
        self._check()
        if self._busy or self.day != len(self.times):
            raise RuntimeError("两钱包尚未完成全部行")
        spot, usdm = self._s.result(), self._u.result()
        if np.any(spot.cash_flow + usdm.cash_flow != 0):
            self._invalid = True
            raise RuntimeError("内部划转现金流不守恒")
        equity = self._equity.copy()
        prev = np.r_[sum(self.initial), equity[:-1]]
        returns = np.divide(equity, prev, out=np.ones_like(prev), where=prev > 0) - 1
        residual = self._residual_value.copy()
        gross = np.sum(np.abs(residual), axis=1)
        exceeded = ~np.isfinite(gross) | (gross > self.residual_limit)
        logs = copy.deepcopy(self.transfers)
        liquidation_rows = sorted(set(e.day for e in usdm.liquidation_events))
        for event in logs:
            if event["rescue_failed"]:
                event["next_liquidation_row"] = next(
                    (d for d in liquidation_rows if d >= event["row"]), None
                )
        leg_diags = []
        for side, (m, result) in enumerate(
            ((self.spot_market, spot), (self.usdm_market, usdm))
        ):
            diag = copy.deepcopy(m.diagnostics)
            diag.update(
                periods_per_year=8760,
                daily_periods_per_year=365,
                carry_blocked=self._configs[side].carry_blocked,
                rolls=[],
                roll_overlap_events=0,
                clearance_fee=float(np.sum(result.clearance_fee))
                if result.clearance_fee is not None
                else 0.0,
                bankruptcy_deficit=float(np.sum(result.bankruptcy_deficit))
                if result.bankruptcy_deficit is not None
                else 0.0,
                liquidation_counts={
                    stage: int(result.coverage.get("liquidation_" + stage, 0))
                    for stage in ("open", "post_trade", "intraday", "close")
                },
                tier_cap_exceeded=int(result.coverage.get("tier_cap_exceeded", 0)),
                blocked_buy=int(result.n_blocked_buy.sum()),
                blocked_sell=int(result.n_blocked_sell.sum()),
                filtered_orders={
                    k: int(v)
                    for k, v in result.coverage.items()
                    if "rounded" in k or "notional" in k or "qty" in k
                },
            )
            diag["valid"] = not diag["tier_cap_exceeded"]
            diag["liquidation_events"] = [
                dict(
                    row=e.day,
                    time=int(m.times[e.day]),
                    symbol=m.symbols[e.col],
                    quantity=e.qty,
                )
                for e in result.liquidation_events
            ]
            diag["unrealized"] = [
                dict(
                    symbol=m.symbols[j],
                    quantity=float(self._spot_shares[j]),
                    value=float(self._spot_shares[j] * m.market.close[-1, j]),
                )
                for j in range(len(m.symbols))
                if m.kind == "spot"
                and m.arrays["lifecycle"][-1, j] == "ended"
                and self._spot_shares[j] != 0
            ]
            result.diagnostics["crypto"] = diag
            leg_diags.append(diag)
        truncated = [e for e in logs if e["actual"] < e["requested"]]
        failed = [e for e in logs if e["rescue_failed"]]
        diag = dict(
            spot=leg_diags[0],
            usdm=leg_diags[1],
            transfer_model="未验证的划转模型假设",
            transfer_margin_factor=self.transfer_margin_factor,
            transfers=sum(e["actual"] > 0 for e in logs),
            transfer_requests=len(logs),
            requested_total=sum(e["requested"] for e in logs),
            actual_total=sum(e["actual"] for e in logs),
            truncated_count=len(truncated),
            truncated_amount=sum(e["requested"] - e["actual"] for e in truncated),
            rejected_deposits=sum(e["reason"] == "receiver_bankrupt" for e in logs),
            rescue_failed_count=len(failed),
            rescue_failed_amount=sum(e["requested"] - e["actual"] for e in failed),
            rescue_failures_followed_by_liquidation=sum(
                e["next_liquidation_row"] is not None for e in failed
            ),
            rounding_residual_max=float(np.abs(self._rounding_value).sum(axis=1).max()),
            underfill_residual_max=float(
                np.abs(self._underfill_value).sum(axis=1).max()
            ),
            residual_max=float(np.max(gross)),
            residual_rows=int(np.count_nonzero(gross > 1e-12)),
            residual_exceeded_rows=int(exceeded.sum()),
            periods_per_year=8760,
            daily_periods_per_year=365,
            valid=not exceeded.any() and all(d["valid"] for d in leg_diags),
        )
        r = TwoWalletResult(
            spot,
            usdm,
            self.times.copy(),
            self.coins,
            equity,
            equity / sum(self.initial),
            returns,
            self._residual.copy(),
            residual,
            self._rounding_value.copy(),
            self._underfill_value.copy(),
            logs,
            diag,
        )
        if exceeded.any() and not self.allow_residual:
            raise InvalidCryptoResult(
                "两钱包残余暴露超过 residual_limit；研究可显式 allow_residual=True", r
            )
        if leg_diags[1]["tier_cap_exceeded"] and not self.allow_tier_exceeded:
            raise InvalidCryptoResult("tier_cap_exceeded；第一档模型无效", r)
        return r

    def snapshot(self):
        self._check()
        if self._busy or self._s.day != self._u.day:
            raise RuntimeError("快照只允许两边完成同一行后创建")
        _strategy(self.transfer, require=True)
        if callable(self.targets):
            _strategy(self.targets, require=True)
        data = dict(
            version=1,
            fingerprint=self._fingerprint,
            day=self.day,
            spot=base64.b64encode(self._s.snapshot()).decode(),
            usdm=base64.b64encode(self._u.snapshot()).decode(),
            transfers=self.transfers,
            ratio=self._ratio.tolist(),
            exit_pending=self._exit_pending.tolist(),
            equity=self._equity[: self.day].tolist(),
            residual=self._residual[: self.day].tolist(),
            residual_value=self._residual_value[: self.day].tolist(),
            rounding_value=self._rounding_value[: self.day].tolist(),
            underfill_value=self._underfill_value[: self.day].tolist(),
            transfer_state=None if self.transfer is None else self.transfer.state(),
            target_state=self.targets.state() if callable(self.targets) else None,
        )
        payload = _json(data)
        return b"TWALLET1" + hashlib.sha256(payload).digest() + payload

    @classmethod
    def restore(cls, snapshot, spot, usdm, targets=None, **kwargs):
        blob = bytes(snapshot)
        if blob[:8] != b"TWALLET1" or hashlib.sha256(blob[40:]).digest() != blob[8:40]:
            raise ValueError("两钱包快照格式或校验和错误")
        data = json.loads(blob[40:])
        required_fields = {
            "version",
            "fingerprint",
            "day",
            "spot",
            "usdm",
            "transfers",
            "ratio",
            "exit_pending",
            "equity",
            "residual",
            "residual_value",
            "rounding_value",
            "underfill_value",
            "transfer_state",
            "target_state",
        }
        if not isinstance(data, dict) or set(data) != required_fields:
            raise ValueError(
                "两钱包快照 schema 字段不完整或含未知字段；不支持旧 TWALLET1"
            )
        # 回调也复制，失败不会修改调用方的策略状态。
        kwargs = dict(kwargs)
        if "transfer" in kwargs:
            kwargs["transfer"] = _clone_strategy(kwargs["transfer"])
        targets = _clone_strategy(targets)
        fresh = cls(spot, usdm, targets, **kwargs)
        if data["version"] != 1 or data["fingerprint"] != fresh._fingerprint:
            raise ValueError(
                "两钱包快照的时间轴、映射、策略或配置指纹不一致；不迁移旧 initial_state 指纹"
            )
        s = bt.Simulator.restore(
            base64.b64decode(data["spot"], validate=True),
            fresh._s._market,
            spot.costs,
            fresh._configs[0],
            kind="shares",
            fingerprint="full",
        )
        u = bt.Simulator.restore(
            base64.b64decode(data["usdm"], validate=True),
            usdm.market,
            usdm.costs,
            fresh._configs[1],
            kind="shares",
            fingerprint="full",
        )
        day = data["day"]
        if s.day != u.day or s.day != day or not 0 <= day <= len(fresh.times):
            raise ValueError("两钱包快照行号不一致")
        for name, key in (
            ("_equity", "equity"),
            ("_residual", "residual"),
            ("_residual_value", "residual_value"),
            ("_rounding_value", "rounding_value"),
            ("_underfill_value", "underfill_value"),
        ):
            if day:
                getattr(fresh, name)[:day] = np.asarray(data[key])
        fresh._ratio[:] = data["ratio"]
        fresh._exit_pending[:] = data["exit_pending"]
        fresh.transfers = copy.deepcopy(data["transfers"])
        if fresh.transfer is not None:
            _strategy(fresh.transfer, require=True)
            fresh.transfer.restore(data["transfer_state"])
        if callable(fresh.targets):
            _strategy(fresh.targets, require=True)
            fresh.targets.restore(data["target_state"])
        fresh._s, fresh._u, fresh._day = s, u, day
        fresh._spot_shares, fresh._usdm_shares = s.state.shares, u.state.shares
        return fresh


def two_wallet(spot, usdm, targets, **kwargs):
    """一次跑完两钱包；参数与 TwoWallet 一致。"""
    return TwoWallet(spot, usdm, targets, **kwargs).run()
