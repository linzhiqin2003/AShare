# 并发测试与性能回归。
#   make test-concurrency                   并发测试（xdist 4 进程、随机顺序）
#   make perf-check                         固定基准与 bench/perfcheck/baseline.json 比较，慢超过 20% 失败
#   make perf-baseline PERFCHECK_NOTE='…'   重录基线（在空闲的机器上录，说明写负载情况）
#   PERFCHECK_ARGS='--only groupbt.single --threads 4' 传给工具的其他参数
# 结果写 .cache/testing/perf.json，供 make test-report 汇总。

PERFCHECK_ARGS ?=
PERFCHECK_NOTE ?=
CONCURRENCY_ARGS ?=

.PHONY: perf-check perf-baseline test-concurrency

perf-check:
	timeout 600 uv run --no-sync python -m bench.perfcheck $(PERFCHECK_ARGS)

perf-baseline:
	timeout 600 uv run --no-sync python -m bench.perfcheck --record --note "$(PERFCHECK_NOTE)" $(PERFCHECK_ARGS)

test-concurrency:
	timeout 900 uv run --no-sync pytest test/concurrency -n 4 -p randomly $(CONCURRENCY_ARGS)
