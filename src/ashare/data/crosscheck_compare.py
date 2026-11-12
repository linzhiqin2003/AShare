"""独立可测试的双源比较；缺失返回与差异分别计数。"""

from .catalog import SCALES


def compare_rows(
    rows, sampled, st, suspended, selected_days, actual_rows, expected_status
):
    differences, price_checks, status_checks, covered = [], 0, 0, set()
    for row in actual_rows:
        key = row["symbol"], row["date"]
        covered.add(key)
        quote = rows.get(key)
        if key in sampled and quote:
            for field in ("open", "high", "low", "close", "volume"):
                expected = (
                    quote["vol"] * SCALES["tushare_volume"]
                    if field == "volume"
                    else quote[field]
                )
                actual = float(row[field]) if row.get(field) else None
                price_checks += 1
                if actual is None or abs(expected - actual) > (
                    1.0 if field == "volume" else 0.011
                ):
                    differences.append(
                        dict(
                            symbol=key[0],
                            date=key[1],
                            field=field,
                            tushare=expected,
                            baostock=actual,
                            judgment="成交额/复权口径需第三方成交记录确认；现有两源不足以裁决",
                        )
                    )
        if key in expected_status:
            for field, expected, actual in (
                ("is_st", key in st, row.get("isST") == "1"),
                ("is_suspended", key in suspended, row.get("tradestatus") == "0"),
            ):
                status_checks += 1
                if expected != actual:
                    judgment = (
                        "源日线有正成交量，支持非全天停牌"
                        if field == "is_suspended" and quote and quote["vol"] > 0
                        else "缺少交易所生效公告，不能仅凭两个源判断谁对"
                    )
                    differences.append(
                        dict(
                            symbol=key[0],
                            date=key[1],
                            field=field,
                            tushare=expected,
                            baostock=actual,
                            judgment=judgment,
                        )
                    )
    missing_samples = [dict(symbol=s, date=d) for s, d in sorted(sampled - covered)]
    missing_status = [
        dict(symbol=s, date=d) for s, d in sorted(expected_status - covered)
    ]
    return dict(
        price_checks=price_checks,
        status_checks=status_checks,
        differences=differences,
        missing_samples=missing_samples,
        missing_status=missing_status,
    )
