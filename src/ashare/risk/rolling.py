"""行情描述子的固定窗口递推；从同一前缀起点归约，追加与全量顺序一致。"""

import numpy as np


class RollingDescriptors:
    def __init__(self, source, returns, config, listing_ages):
        self.source, self.returns, self.config, self.ages = (
            source,
            returns,
            config,
            listing_ages,
        )
        self.day = -1
        n = returns.shape[1]
        self.market = np.zeros(len(returns))
        self.market_valid = np.zeros(len(returns), bool)
        self.beta_sum = np.zeros((6, n))
        self.beta_count = np.zeros(n, dtype=np.int64)
        self.vol_sum = np.zeros((3, n))
        self.vol_count = np.zeros(n, dtype=np.int64)
        self.momentum_sum = np.zeros((2, n))
        self.momentum_count = np.zeros(n, dtype=np.int64)
        self.turnover_sum = np.zeros((3, n))
        self.turnover_count = np.zeros((3, n), dtype=np.int64)
        self.cmra_sum = np.zeros(n)
        self.cmra_count = np.zeros(n, dtype=np.int64)
        self.cmra_prefix = np.zeros((253, n))
        self.cmra_valid_prefix = np.zeros((253, n), dtype=np.int64)

    def _return(self, day, market=False):
        n = self.returns.shape[1]
        if day < 1:
            return np.zeros((6 if market else 3, n)), np.zeros(n, bool)
        y = self.returns[day]
        valid = np.isfinite(y)
        if market:
            valid &= self.market_valid[day]
        y = np.where(valid, y, 0.0)
        if market:
            x = self.market[day]
            return np.array(
                [valid.astype(float), valid * x, y, valid * x * x, y * x, y * y]
            ), valid
        return np.array([valid.astype(float), y, y * y]), valid

    def _momentum(self, day):
        n = self.returns.shape[1]
        if day < 1:
            return np.zeros((2, n)), np.zeros(n, bool)
        y = self.returns[day]
        valid = np.isfinite(y)
        try:
            rf = self.source["risk_free"][day]
        except KeyError:
            rf = 0.0
        value = np.log1p(np.maximum(y + rf, -0.999999999)) - np.log1p(rf)
        return np.array([valid.astype(float), np.where(valid, value, 0.0)]), valid

    def _turnover(self, day):
        n = self.returns.shape[1]
        if day < 0:
            return np.zeros(n), np.zeros(n, bool)
        x = self.source["turnover_rate"][day]
        valid = self.source["is_listed"][day].astype(bool) & np.isfinite(x) & (x >= 0)
        return np.where(valid, x, 0.0), valid

    @staticmethod
    def _update(sums, counts, new, old, decay, window):
        sums *= decay
        sums += new[0]
        sums -= decay**window * old[0]
        counts += new[1].astype(np.int64) - old[1].astype(np.int64)

    def at(self, day):
        if day < self.day:
            # 历史按需查询可重新播放前缀；不使用未来的递推状态。
            self.__init__(self.source, self.returns, self.config, self.ages)
        c = self.config
        for t in range(self.day + 1, day + 1):
            if t >= 1:
                cap = self.source["total_mv"][t - 1]
                ok = (
                    np.isfinite(self.returns[t])
                    & np.isfinite(cap)
                    & (cap > 0)
                    & self.source["is_listed"][t - 1].astype(bool)
                    & ~self.source["is_suspended"][t - 1].astype(bool)
                    & (self.ages[t - 1] >= c.min_listing_days)
                )
                weights = np.where(ok, cap, 0.0)
                total = weights.sum()
                self.market_valid[t] = total > 0
                self.market[t] = (
                    (weights * np.nan_to_num(self.returns[t])).sum() / total
                    if total > 0
                    else 0.0
                )
            self._update(
                self.beta_sum,
                self.beta_count,
                self._return(t, True),
                self._return(t - c.beta_window, True),
                2 ** (-1 / c.beta_half_life),
                c.beta_window,
            )
            self._update(
                self.vol_sum,
                self.vol_count,
                self._return(t),
                self._return(t - 252),
                2 ** (-1 / 42),
                252,
            )
            log_values, log_valid = self._momentum(t)
            log_valid = log_valid & np.isfinite(log_values[1])
            self.cmra_sum += np.where(log_valid, log_values[1], 0.0)
            self.cmra_count += log_valid.astype(np.int64)
            self.cmra_prefix[t % 253] = self.cmra_sum
            self.cmra_valid_prefix[t % 253] = self.cmra_count
            m = t - c.momentum_skip
            self._update(
                self.momentum_sum,
                self.momentum_count,
                self._momentum(m),
                self._momentum(m - c.momentum_window),
                2 ** (-1 / c.momentum_half_life),
                c.momentum_window,
            )
            new, valid = self._turnover(t)
            for j, window in enumerate((21, 63, 252)):
                old, old_valid = self._turnover(t - window)
                self.turnover_sum[j] += new - old
                self.turnover_count[j] += valid.astype(np.int64) - old_valid.astype(
                    np.int64
                )
        self.day = day
        w, sx, sy, sxx, sxy, syy = self.beta_sum / np.maximum(self.beta_sum[0], 1e-30)
        vx = np.maximum(sxx - sx * sx, 0.0)
        cov = sxy - sx * sy
        beta = np.divide(cov, vx, out=np.zeros_like(cov), where=vx > 1e-20)
        hsigma = np.sqrt(np.maximum(syy - sy * sy - beta * cov, 0.0))
        beta[self.beta_count < c.min_descriptor_days] = np.nan
        hsigma[self.beta_count < c.min_descriptor_days] = np.nan
        _, vmean, vsquare = self.vol_sum / np.maximum(self.vol_sum[0], 1e-30)
        dastd = np.sqrt(np.maximum(vsquare - vmean * vmean, 0.0))
        dastd[self.vol_count < c.min_descriptor_days] = np.nan
        momentum = self.momentum_sum[1] / np.maximum(self.momentum_sum[0], 1e-30)
        momentum[self.momentum_count < c.min_descriptor_days] = np.nan
        cmra = np.full(self.returns.shape[1], np.nan)
        if day >= 252:
            # 共同的当日累计项在极差中抵消，只需十二个回看端点。
            endpoints = self.cmra_prefix[(day - np.arange(1, 13) * 21) % 253]
            enough = self.cmra_count - self.cmra_valid_prefix[(day - 252) % 253] == 252
            cmra[enough] = np.ptp(endpoints[:, enough], axis=0)
        return {
            "cmra": cmra,
            "beta": beta,
            "hsigma": hsigma,
            "dastd": dastd,
            "momentum": momentum,
            "turnover_sum": self.turnover_sum.copy(),
            "turnover_count": self.turnover_count.copy(),
        }

    def checkpoint(self):
        """仅保留递推量和市场收益序列，不复制行情面板。"""
        state = {
            name: getattr(self, name).copy()
            for name in (
                "beta_sum",
                "beta_count",
                "vol_sum",
                "vol_count",
                "momentum_sum",
                "momentum_count",
                "turnover_sum",
                "turnover_count",
                "cmra_sum",
                "cmra_count",
                "cmra_prefix",
                "cmra_valid_prefix",
            )
        }
        state.update(
            day=np.array(self.day),
            market=self.market[: self.day + 1].copy(),
            market_valid=self.market_valid[: self.day + 1].copy(),
        )
        return state

    def restore(self, state):
        day = int(state["day"])
        if not -1 <= day < len(self.returns):
            raise ValueError("递推检查点超出输入日期范围")
        for name, value in state.items():
            if name in ("market", "market_valid"):
                getattr(self, name)[: len(value)] = value
            elif name != "day":
                if getattr(self, name).shape != value.shape:
                    raise ValueError("递推检查点证券维度不同")
                setattr(self, name, value.copy())
        self.day = day
