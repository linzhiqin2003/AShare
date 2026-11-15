"""Independent C4 oracle and required random branch coverage."""

from test.generators_wallets import corpus


def test_c4_independent_random_corpus_all_branches():
    corpus()


def test_c4_decimal_multicoin_daily_delisting_corpus_all_branches():
    from test.generators_wallets import decimal_daily_corpus

    decimal_daily_corpus()
