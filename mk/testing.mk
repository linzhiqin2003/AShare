# 测试指标：分类运行、覆盖率、耗时、flaky、汇总。用法见 docs/TESTING.md。
# WORKERS 默认取 bench/testing/budgets.py 里的值；WORKERS=0 串行。

TESTING_PY := uv run --no-sync python bench/testing
WORKERS ?=
N ?=
CAT ?=
SEED ?=
_workers = $(if $(WORKERS),--workers $(WORKERS))

.PHONY: test test-serial test-cat coverage coverage-python coverage-rust flaky test-report test-metrics

# 全量并行跑，记录耗时（.cache/testing/runtime.json）。
test:
	$(TESTING_PY)/runtime.py $(_workers)

# 全量串行跑一遍，记录串行耗时（.cache/testing/runtime-serial.json）。
test-serial:
	$(TESTING_PY)/runtime.py --workers 0

# 按类别跑：make test-cat CAT=property（多个类别用 pytest -m 表达式，如 CAT="unit or property"）。
test-cat:
	@test -n "$(CAT)" || { echo "usage: make test-cat CAT=unit|integration|e2e|smoke|concurrency|perf|property|differential"; exit 2; }
	$(TESTING_PY)/runtime.py $(_workers) --category "$(CAT)"

coverage:
	$(TESTING_PY)/coverage_run.py $(_workers)

coverage-python:
	$(TESTING_PY)/coverage_run.py $(_workers) --skip-rust

coverage-rust:
	$(TESTING_PY)/coverage_run.py --skip-python

flaky:
	$(TESTING_PY)/flaky.py $(_workers) $(if $(N),--runs $(N)) $(if $(SEED),--seed $(SEED))

test-report:
	$(TESTING_PY)/report.py

# 一次跑齐覆盖率、耗时、flaky 再汇总（变异和性能由各自的目标产出）。
test-metrics:
	-$(MAKE) test
	-$(MAKE) coverage
	-$(MAKE) flaky
	$(MAKE) test-report
