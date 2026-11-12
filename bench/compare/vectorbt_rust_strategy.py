"""Optional open-source vectorbt-rust engine using the identical strategy."""

import vectorbt_strategy as strategy


def run(market, target):
    strategy.ENGINE = "rust"
    return strategy.run_orders(market, target)


def batch(market, targets):
    strategy.ENGINE = "rust"
    return strategy.batch(market, targets)


def prepare(market, target):
    p = market.exec_price
    sizes = target.to_dense(*p.shape)
    return (
        p,
        sizes,
        strategy.exit_first_sequence(sizes, target.days, p.shape[1]),
        len(target.weights),
    )


def execute(prepared):
    strategy.ENGINE = "rust"
    return strategy.execute_orders(prepared)


def prepare_batch(market, targets):
    return strategy.prepare_batch(market, targets)


def execute_batch(chunks):
    strategy.ENGINE = "rust"
    return strategy.execute_batch(chunks)
