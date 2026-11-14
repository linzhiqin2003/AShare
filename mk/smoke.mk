# 冒烟、集成、端到端测试（docs/TESTING_PLAN.md）。全部只用合成数据（test/fixtures/synthetic_datadir.py），
# 不联网、不读 data/。合成数据目录按代码哈希缓存在 .cache/test-datadir/，第一次运行时生成。
#
#   make smoke             导入全部模块、每个命令行入口 --help、平台起停一遍（1 分钟内）
#   make test-integration  数据层 → 因子 → 分组回测 → 报告；平台 HTTP → 工作进程 → 落盘 → 读回
#   make test-e2e          命令行与平台 API 各跑一次完整研究
#   make test-workflows    以上三项
#
# 共享机器上 xdist 最多 4 个进程；每条命令都有总超时（pytest-timeout 管单个测试）。

SMOKE_PYTEST = uv run --no-sync pytest -q -p no:randomly -W ignore::DeprecationWarning --durations=5

.PHONY: smoke test-integration test-e2e test-workflows

smoke:
	$(SMOKE_PYTEST) --timeout=120 test/smoke

test-integration:
	$(SMOKE_PYTEST) --timeout=600 -n 4 test/integration

test-e2e:
	$(SMOKE_PYTEST) --timeout=600 -n 2 test/e2e

test-workflows: smoke test-integration test-e2e
