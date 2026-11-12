"""证券档案：这里只表达申报单位和报价单位，完整费用规则留给 P3。"""

from paperbroker import Costs, Instrument, Lots, MarketConfig


def instrument(symbol: str) -> Instrument:
    if symbol.startswith(("688", "689")) and symbol.endswith(".SH"):
        return Instrument(lots=Lots(step=1, minimum=200), profile_id="star-v1")
    if symbol.endswith(".BJ"):
        return Instrument(lots=Lots(step=1, minimum=100), profile_id="bse-v1")
    if (
        symbol.endswith(".SH") and symbol.startswith(("51", "52", "56", "58", "50"))
    ) or (symbol.endswith(".SZ") and symbol.startswith(("15", "16"))):
        return Instrument(tick=10, profile_id="fund-v1", costs=Costs(sell_tax_ppm=0))
    return Instrument()


def configuration(symbols: list[str]) -> MarketConfig:
    return MarketConfig(instruments={s: instrument(s) for s in symbols})


def from_native(config: dict) -> tuple[float, MarketConfig]:
    config = config.copy()
    cash = config.pop("initial_cash") / 10_000
    config.pop("symbols")
    profiles = {}
    for raw in config.pop("instruments"):
        profile = raw.copy()
        name = profile.pop("name")
        profile["costs"], profile["lots"] = (
            Costs(**profile["costs"]),
            Lots(**profile["lots"]),
        )
        profiles[name] = Instrument(**profile)
    config["costs"], config["lots"] = Costs(**config["costs"]), Lots(**config["lots"])
    return cash, MarketConfig(**config, instruments=profiles)
