# 变异测试统一入口，用法见 docs/TESTING_MUTATION.md。
# Python 用 mutmut，Rust 用 cargo-mutants（经 bench/mutate/cargo-shim 重建扩展再跑 pytest）。
# 结果：.cache/testing/mutation.json（分数）、mutation_survivors.md（存活清单）。
# 注意：Makefile 里的 `mutate` 是旧的 bench/mutations.py，这里的单模块目标叫 mutate-module。

# 驱动自己也会 nice 15；这里再包一层，uv 本身也不抢 CPU。JOBS 是同时跑的变异体数，共享机器上最多 2。
MUTATE := nice -n 15 uv run --no-sync python -m bench.mutate.run
JOBS ?= 2
BASE ?= origin/main
MUTATE_TIMEOUT ?= 3600
MUTATE_ARGS ?=

.PHONY: mutate-all mutate-module mutate-diff mutate-report mutate-estimate mutate-controls mutate-legacy

# 全量：所有分片依次跑（约数十小时，见 docs/TESTING_MUTATION.md 的估计）；MUTATE_LANG=python|rust 只跑一种
mutate-all:
	$(MUTATE) full --jobs $(JOBS) --timeout $(MUTATE_TIMEOUT) $(if $(MUTATE_LANG),--language $(MUTATE_LANG)) $(MUTATE_ARGS)

# 单个分片或路径：make mutate-module MODULE=src/ashare/platform/config.py（逗号分隔可给多个）
mutate-module:
	@test -n "$(MODULE)" || { echo "MODULE=<分片名或路径> 必填，分片见 bench/mutate/shards.toml"; exit 2; }
	$(MUTATE) full --module "$(MODULE)" --jobs $(JOBS) --timeout $(MUTATE_TIMEOUT) $(MUTATE_ARGS)

# 只变异相对 BASE 改动过的函数（Python）和改动过的行（Rust），给 PR 用
mutate-diff:
	$(MUTATE) diff --base $(BASE) --jobs $(JOBS) --timeout $(MUTATE_TIMEOUT) $(MUTATE_ARGS)

# 改了 bench/mutate/equivalent.toml 之后，用已存的结果重新出分数和存活清单
mutate-report:
	$(MUTATE) report --scope $(or $(SCOPE),full)

mutate-estimate:
	$(MUTATE) estimate

# 对照实验：变异 0、必死变异、旧扩展混入（Python 部分需要先跑过对应分片）
mutate-controls:
	$(MUTATE) controls $(if $(MUTATE_LANG),--language $(MUTATE_LANG))

# 旧的手写变异脚本：make mutate-legacy LEGACY="rules groupbt"；不给 LEGACY 只列出状态
mutate-legacy:
	$(if $(LEGACY),nice -n 15 uv run --no-sync python -m bench.mutate.legacy run $(LEGACY),uv run --no-sync python -m bench.mutate.legacy list)
