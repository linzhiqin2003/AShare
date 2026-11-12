import numpy as np

from ashare.events.market import BacktestMarket


def test_costs_cover_dates_before_fee_schedule():
    dates = np.array(["20070104", "20080501", "20080919", "20230828"])
    costs = BacktestMarket(None, dates, np.array(["A"]), np.array([0]), None).costs(
        slippage=0.0
    )
    np.testing.assert_allclose(costs.tax_sell, [0.003, 0.001, 0.001, 0.0005])
    assert costs.fee_rate_buy == 1e-5 and costs.slippage == 0.0
