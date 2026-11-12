"""ashare.risk.from_panel：字段改名、行业重编号和命名；与旧手写转接逐位相同。"""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ashare.portfolio import Constraints, PortfolioProblem
from ashare.risk import (
    FundamentalConfig,
    MemoryPanel,
    ModelCache,
    from_panel,
    model_series,
)
from ashare.risk.panel import industry_codes, industry_names

SW = [801780, 801010, 801020, 801030]


class FakePanel:
    def __init__(self, arrays, t, n):
        self._arrays = arrays
        self.dates = np.array(
            [f"2020{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(t)]
        )
        self.symbols = np.array([f"{600000 + j:06d}.SH" for j in range(n)])
        self.version = "fake-v1"

    def __getitem__(self, name):
        return self._arrays[name]


class FakeData:
    def __init__(self, panel, table=True):
        self._panel, self._table, self.requests = panel, table, []

    def panel(self, names, start=None, end=None, symbols=None):
        self.requests.append((tuple(names), start, end, symbols))
        return self._panel

    def table(self, name):
        if not self._table:
            raise KeyError(name)
        assert name == "index_classify"
        return pl.DataFrame(
            {
                "industry_code": [
                    "801010.SI",
                    "801010.SI",
                    "801020.SI",
                    "801030.SI",
                    "801030.SI",
                    "801780.SI",
                    "801780.SI",
                    "801780.SI",
                ],
                "industry_name": [
                    "农林牧渔",
                    "农林牧渔",
                    "采掘",
                    "化工",
                    "基础化工",
                    "银行",
                    "银行",
                    "银行II",
                ],
                "classification": [
                    "SW2014",
                    "SW2021",
                    "SW2014",
                    "SW2014",
                    "SW2021",
                    "SW2014",
                    "SW2021",
                    "SW2021",
                ],
                "level": ["L1", "L1", "L1", "L1", "L1", "L1", "L1", "L2"],
            }
        ).lazy()


def panel(t=90, n=60, seed=71):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, (t, n)), axis=0))
    industry = np.broadcast_to(np.array(SW)[np.arange(n) % 4], (t, n)).astype(float)
    industry[:, :3] = 0.0  # 缺失
    arrays = dict(
        close=close,
        adj_factor=np.ones((t, n)),
        total_market_cap=np.exp(rng.normal(20, 1, (t, n))),
        float_market_cap=np.full((t, n), 1e9),
        turnover=rng.uniform(0.001, 0.05, (t, n)).astype(np.float32),
        pb=rng.uniform(0.5, 5, (t, n)).astype(np.float32),
        pe_ttm=rng.uniform(5, 40, (t, n)).astype(np.float32),
        industry=industry,
        is_suspended=np.zeros((t, n), bool),
        is_listed=np.ones((t, n), bool),
    )
    arrays["is_listed"][:70, 5] = False
    arrays["pb"][30:45, 6] = np.nan
    return FakePanel(arrays, t, n)


def old_adapter(p):
    """bench/e2e/risk_adapter.py 改用正式入口之前的写法。"""
    names = {
        "close": "close",
        "adj_factor": "adj_factor",
        "total_mv": "total_market_cap",
        "circ_mv": "float_market_cap",
        "turnover_rate": "turnover",
        "pb": "pb",
        "pe_ttm": "pe_ttm",
        "is_suspended": "is_suspended",
        "is_listed": "is_listed",
    }
    fields = {}
    for target, name in names.items():
        value = p[name]
        fields[target] = (
            value.astype(bool)
            if target.startswith("is_")
            else np.asarray(value, dtype=np.float64)
        )
    raw = np.asarray(p["industry"], dtype=np.int64)
    codes = np.unique(raw[raw > 0])
    lookup = np.full(int(codes.max()) + 1, -1, dtype=np.int64)
    lookup[codes] = np.arange(len(codes))
    fields["industry"] = lookup[raw]
    source = MemoryPanel(p.dates, tuple(p.symbols), p.version, fields)
    return source, FundamentalConfig(industry_codes=tuple(range(len(codes))))


SMALL = dict(
    min_listing_days=15,
    min_descriptor_days=10,
    structural_days=30,
    history=60,
    beta_window=30,
    momentum_window=30,
    momentum_skip=5,
)


def test_fields_and_industries():
    p = panel()
    data = FakeData(p)
    source, config = from_panel(data, start="20200101")
    assert data.requests[0][1] == "20200101"
    assert source["turnover_rate"].dtype == np.float64
    assert source["is_listed"].dtype == bool
    assert config.industry_codes == (0, 1, 2, 3)
    # 升序代码 801010 801020 801030 801780；SW2021 名称优先，只有旧版的用旧版。
    assert config.industry_names == ("农林牧渔", "采掘", "基础化工", "银行")
    assert source["industry"][0, :5].tolist() == [-1, -1, -1, 2, 3]


def test_names_fall_back_to_codes_without_table():
    _, config = from_panel(FakeData(panel(), table=False))
    assert config.industry_names == ("801010", "801020", "801030", "801780")


def test_industry_codes_nan_and_zero_are_missing():
    out, codes = industry_codes(np.array([[0.0, np.nan, 801780.0, 801010.0]]))
    assert out.tolist() == [[-1, -1, 1, 0]] and codes.tolist() == [801010, 801780]


def test_duplicate_names_get_code_suffix():
    class D:
        def table(self, name):
            return pl.DataFrame(
                {
                    "industry_code": ["1.SI", "2.SI"],
                    "industry_name": ["甲", "甲"],
                    "classification": ["SW2021", "SW2021"],
                    "level": ["L1", "L1"],
                }
            ).lazy()

    assert industry_names(D(), np.array([1, 2])) == ("甲_1", "甲_2")


def test_config_overrides_and_validation():
    _, config = from_panel(
        FakeData(panel()), history=60, config=FundamentalConfig(shrinkage=0.2)
    )
    assert config.history == 60 and config.shrinkage == 0.2
    with pytest.raises(ValueError, match="industry_names"):
        FundamentalConfig(industry_codes=(0, 1), industry_names=("a",))
    with pytest.raises(ValueError, match="industry_names"):
        FundamentalConfig(industry_codes=(0, 1), industry_names=("a", "a"))
    with pytest.raises(ValueError, match="industry_names"):
        FundamentalConfig(industry_codes=(0,), industry_names=("unknown",))
    with pytest.raises(ValueError, match="start/end/symbols"):
        from_panel(FakeData(panel()), panel=panel(), start="20200101")


def test_matches_old_adapter_bit_for_bit_except_factor_names(tmp_path):
    p = panel()
    old_source, old_config = old_adapter(p)
    source, config = from_panel(FakeData(p))
    for name in old_source.fields:
        a, b = source[name], old_source[name]
        assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), name
    old_config = replace(old_config, **SMALL)
    config = replace(config, **SMALL)
    new_models = list(model_series(source, config=config))
    old_models = list(model_series(old_source, config=old_config))
    assert len(new_models) == len(old_models) == 90
    for a, b in zip(new_models, old_models, strict=True):
        for k in ("B", "F", "D"):
            assert np.array_equal(getattr(a, k), getattr(b, k)), (a.date, k)
    last = new_models[-1]
    assert last.factors[1:6] == (
        "industry_农林牧渔",
        "industry_采掘",
        "industry_基础化工",
        "industry_银行",
        "industry_unknown",
    )
    assert old_models[-1].factors[1:5] == (
        "industry_0",
        "industry_1",
        "industry_2",
        "industry_3",
    )
    # 名字不同的配置不会命中对方的缓存。
    cache = tmp_path / "cache"
    first = list(model_series(old_source, config=old_config, cache=ModelCache(cache)))
    again = list(model_series(source, config=config, cache=ModelCache(cache)))
    assert (
        first[-1].factors[1] == "industry_0"
        and again[-1].factors[1] == "industry_农林牧渔"
    )


def test_portfolio_accepts_factor_names():
    p = panel()
    source, config = from_panel(FakeData(p))
    model = list(model_series(source, config=replace(config, **SMALL)))[-1]
    n = len(model.D)
    industries = [f for f in model.factors if f.startswith("industry_")]
    styles = ("size", "beta")
    by_name = PortfolioProblem(
        alpha=np.zeros(n),
        risk=model,
        benchmark=np.full(n, 1 / n),
        current=np.zeros(n),
        constraints=Constraints(industry_bound=0.02, style_bound=0.1, max_weight=0.1),
        industry_factors=industries,
        style_factors=styles,
    )
    idx = [model.factors.index(f) for f in industries]
    by_index = PortfolioProblem(
        alpha=np.zeros(n),
        risk=model,
        benchmark=np.full(n, 1 / n),
        current=np.zeros(n),
        constraints=Constraints(industry_bound=0.02, style_bound=0.1, max_weight=0.1),
        industry_factors=tuple(idx),
        style_factors=(model.factors.index("size"), model.factors.index("beta")),
    )
    assert by_name.industry_factors == by_index.industry_factors == tuple(idx)
    np.testing.assert_array_equal(by_name.industries, by_index.industries)
    np.testing.assert_array_equal(by_name.styles, by_index.styles)
    mixed = PortfolioProblem(
        alpha=np.zeros(n),
        risk=model,
        benchmark=np.full(n, 1 / n),
        current=np.zeros(n),
        style_factors=("size", model.factors.index("beta")),
    )
    assert mixed.style_factors == by_index.style_factors
    with pytest.raises(ValueError, match="没有因子 'nope'"):
        PortfolioProblem(
            alpha=np.zeros(n),
            risk=model,
            benchmark=np.full(n, 1 / n),
            current=np.zeros(n),
            style_factors=("nope",),
        )
