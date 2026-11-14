# 测试指标

`make test-report` 生成，每天一节，同一天重跑会覆盖当天那一节。各指标的口径见
`docs/TESTING.md`；机器可读版在 `.cache/testing/report.json`（不入库）。

## 2026-11-14

| 指标 | 结果 |
|---|---|
| Branch coverage（Python） | 84.0%（8327/9918 个分支；行 91.8%；66 个文件低于 80%）（2026-11-14 13:53） |
| Coverage（Rust 行覆盖，stable 无分支数据） | 34.6%，仅 `cargo test` 范围，10 个文件低于目标 |
| Test runtime | 并行 327 s（6 进程，4612 个测试，测试耗时合计 1447 s）；串行 818 s；103 个超出单测预算（串行计）（2026-11-14 13:53） |
| Mutation score | 未运行 |
| Flaky rate | 0.000%（0/4612，3 轮随机顺序；一直失败 0 个）（2026-11-14 14:13） |
| 性能回归 | 未运行 |

分支覆盖最低的文件：

- `src/ashare/analysis/bench.py` 0.0%
- `src/ashare/analysis/example.py` 0.0%
- `src/ashare/data/futures_hedge_check.py` 0.0%
- `src/ashare/data/sources/baostock.py` 0.0%
- `src/ashare/events/bench.py` 0.0%
- `src/ashare/events/examples.py` 0.0%
- `src/ashare/report/context.py` 14.3%
- `src/ashare/paper/verify_live.py` 21.4%
- `src/ashare/report/__init__.py` 35.7%
- `src/ashare/data/crosscheck.py` 37.5%

各类别（测试耗时合计）：differential 1910 个 192 s、property 8 个 2 s、unit 2694 个 1253 s

最慢的 5 个：

- 57.7 s `test.platform.test_storage::test_filtered_counts_groups_and_page_use_same_population`
- 25.7 s `packages.paperbroker.tests.test_wide::test_wide_random_differential`
- 18.2 s `test.factor.test_mining_search::test_planted_signal_is_found`
- 16.8 s `test.test_extended::test_legacy_bitwise_fixture`
- 16.6 s `test.data.test_hk_data::test_cli_status_and_verify_exit_code`
