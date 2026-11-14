# 大模型研究循环（`ashare.factor.mining.llm`）

`LLMResearch` 是 `mine()` 的第四种搜索策略：不随机生成表达式，而是让大模型按「提出假设 → 写成因子 → 看训练段回测反馈 → 记笔记 → 下一轮」的循环提出候选。评估、去重、台账、多重检验、留出段规则全部沿用 `mine()`（`docs/DESIGN.md` 6.2.2），这里只决定「下一批试哪些表达式」。

## 与 RD-Agent 的关系

思路来自微软 [RD-Agent](https://github.com/microsoft/RD-Agent)（MIT 许可，论文 *R&D-Agent-Quant*，NeurIPS 2025）的量化研究循环：假设生成 → 把假设拆成具体的因子任务 → 实现 → 运行评估 → 与当前最好的比较、给出反馈 → 知识沉淀 → 下一轮。我们没有依赖它，而是在本框架里重写：

| RD-Agent | 这里 |
|---|---|
| 模型写 Python 因子代码，在沙箱里运行 | 模型只写本仓库的表达式语言，`parse` 按搜索空间白名单构造 `Expr`，不执行任何代码 |
| 自己的回测与指标 | `mine()` 的评估器：训练段粗筛（Rank IC 的 t 值）+ 多头端回测，适应度只用训练段 |
| 知识库 | 模型自己维护的一段研究笔记（纯文本，有字数上限） |
| 调度器在探索与细化之间选择 | 轻量版：连续 `stall` 轮没有提高最好适应度时建议「探索新假设」，否则建议「细化当前最好的」；模型可以不采纳，在 `action` 里写明 |

提示词是自己写的，没有复制 RD-Agent 的提示词文本。

## 用法

```python
from ashare.factor.mining import LLMResearch, Split, mine
from ashare.factor.mining.llm import ChatClient
from ashare.factor.mining.synthetic import market

panel = market(t=900, n=150, strength=0.003)                 # 合成行情，埋了一个已知信号
split = Split.recent(panel.dates, holdout_years=1, valid_years=1)
strategy = LLMResearch(ChatClient(), rounds=10, batch=8, budget_usd=1.0)
result = mine(panel, split=split, strategy=strategy)          # 台账、转写记录在 default_root()
print(result.text())
print(strategy.last_run)        # 轮数、停止原因、token、费用、转写记录路径
```

命令行（真实数据用 `ashare.data.open()` 的面板与 `Split.standard()`，`--synthetic` 用合成行情）：

```bash
python -m ashare.factor.mining.llm --rounds 20 --batch 8 --budget-usd 2
python -m ashare.factor.mining.llm --synthetic --rounds 5 --budget-usd 0.2
python -m ashare.factor.mining.llm --screen gpu ...           # GPU 粗筛（可选依赖 gpu）
```

其余选项：`--model`、`--base-url`、`--key-env`、`--temperature`、`--max-tokens`（每次回答的输出上限，含思考，默认 32768）、`--reasoning off|low|medium|high`（思考强度，只支持 OpenRouter；不传用模型默认）、`--price-in/--price-cached/--price-out`、`--budget-tokens`、`--patience`、`--root`、`--screen-t`、`--start`、`--threads`。

## 模型接口

`ChatClient` 是一个只用 httpx 的 OpenAI 兼容客户端，默认 DeepSeek 官方接口（`https://api.deepseek.com`，模型 `deepseek-flash`），密钥读环境变量 `DEEPSEEK_API_KEY`（`api_key_env=` 可改）。

- 密钥第一次请求时才读，不写日志、不进转写记录，任何报错消息都先把密钥替换成 `***`；
- 要求 JSON 输出（`response_format: json_object`），不用流式；`timeout` 管整次请求（连接、等响应头、读完正文）；
- 429、5xx、传输错误按指数退避重试（有 `Retry-After` 时取较大者），重试用完或遇到其他 4xx 抛 `LLMError`，消息只有状态码和接口的简短说明；
- `seed` 只在 `send_seed=True` 时放进请求（不是所有兼容接口都接受）；
- 费用按每次 200 响应里的 `usage` 算：缓存未命中的输入、命中缓存的输入（DeepSeek 的 `prompt_cache_hit_tokens`，OpenAI 的 `prompt_tokens_details.cached_tokens`）、输出各一个单价（`Pricing`，每百万 token 美元）。**默认单价是 deepseek-flash 的高峰价（2026-11-14 核对：未命中 0.3、命中 0.006、输出 1.2；非高峰减半），预算按它算只会偏高；换模型或价目变动时显式传入**；响应没有 `usage` 时按字符数估一个偏高的值并标记 `estimated`。

测试注入假客户端（`complete(messages, *, json_mode, seed)` 返回 `Reply`，有 `model` 和 `describe()`），或给 `ChatClient` 传 `transport=httpx.MockTransport(...)`；测试不联网。

## 每一轮

1. **拼提示词**（`prompt.build_messages`）：
   - 系统消息：角色、工作方式、只用给定的字段算子、只输出 JSON；
   - 场景：A 股日频截面选股、多头端检验口径、这次 `mine` 用的适应度（`fitness`、`complexity`）和粗筛门槛 `screen_t`、每个提交都是一次试验；
   - 空间：字段（含量纲）、算子（含允许的参数取值和量纲规则）、结构约束（深度、节点数、根无量纲、截面算子的参数无量纲、不许冗余嵌套）；
   - 语法和 4 个从当前空间里挑出的合法例子；
   - 历史：训练段上最好的 `best_k` 个因子（适应度、训练段 IC 均值与 t 值、训练段多头端信息比率与超额）、最近 `history` 轮的假设与结果、更早提交过的其他表达式（最多 `recent` 条，避免重复）、上一轮的逐条结果（含校验错误、重复、多给被忽略的）；
   - 研究笔记（模型上一轮写的）；
   - 本轮任务：探索或细化的建议、要几个表达式、JSON 格式。
2. **调模型**，回答约定为 `{"action", "hypothesis", "rationale", "factors": [{"name", "expression", "idea"}], "notes"}`。容忍 ```` ```json ```` 围栏和前后多余文字；解析不了的整轮作废，错误写进下一轮的提示词（这一轮照样计费、计轮数）。
3. **评估前校验**：每个表达式用 `parse(text, space)` 构造并按空间检查（白名单算子与字段、参数取值、量纲、深度、节点数、截面位置、冗余），不合法的不评估、不算试验，错误原样反馈；与本次搜索已提交过的去重键（`dedup_key`，含可交换、`ts_corr` 对称、根上的截面单调变换）相同的也不评估；每轮只取前 `batch` 个。
4. **评估**：合法的新候选一批交给 `evaluate`，拿回训练段适应度（粗筛没过、求值失败为 −∞）。
5. **反馈与笔记**：更新最好列表、轮次摘要和笔记（超出 `notes_limit` 截断，模型没给新笔记时保留旧的）；写一行转写记录。

## 停止条件

- 轮数用完（`rounds`）；
- 累计费用达到 `budget_usd` 或 token 达到 `budget_tokens`：在下一轮开始前判断，最多超出一轮的用量（单轮输出受 `max_tokens` 限制）；
- 连续 `patience` 轮没有可评估的候选（回答解析不了、全不合法或全是重复）；
- 模型接口报错（`LLMError`）：停止循环，已评估的候选照常记账、出结果，错误在 `last_run["error"]`。

## 信息隔离：只有训练段

`mine` 把留出段整段截掉（`Truncated`），验证段只用来挑选；策略只能通过 `evaluate` 的返回值（训练段适应度）和两个只读入口拿信息：

- `evaluate.task`：适应度名称、复杂度惩罚、粗筛门槛、`mine` 的台账目录（目录只用来放转写记录，不进提示词）；
- `evaluate.train_feedback(expr)`：`search._train_feedback` 只读结果行的 `train` 段，给出阶段、训练段 IC 均值与 t 值、训练段多头端信息比率与年化超额、是否与已入选因子数值重复。

策略这边再过一道白名单（`prompt.train_only`：只留这几个键、数值必须有限），提示词里不出现任何日期。其他策略不读这两个入口，`mine` 的结果与加入之前逐位相同（冻结摘要测试守住）。

测试（`test/factor/test_mining_llm.py`）：改动验证段和留出段的行情后，每一轮提示词逐字不变，而验证段指标确实变了；改动训练段的行情，提示词随之改变（证明比较是敏感的）；提示词里没有训练段结束以后的日期，也没有验证段的信息比率。

## 身份与可复现

`describe()` 进入搜索身份和台账记录：模型、客户端设置（地址、温度、`max_tokens`、单价，不含密钥）、模板版本 `TEMPLATE_VERSION` 与全部模板文本的哈希 `prompt_sha`、轮数、批量、种子、预算、各项上限。改了提示词，`prompt_sha` 就变。

模型的回答不是确定性的，同样的设置重跑会提出不同的候选；台账照常按 `trial_id` 去重和累计（同一个表达式在不同搜索里只算一次试验），折算显著性用的是台账累计的试验数。

## 转写记录

每次运行一个 JSONL 文件，默认 `<mine 的 root>/llm/<时间>-<身份>.jsonl`（`root` 默认是数据根目录下不入库的 `mining/`），也可以用 `transcript=` 指定。第一行 `start`（策略身份、适应度设置、空间摘要），每轮一行 `round`（提示词哈希、建议、原始回答、用量与费用、假设、理由、笔记、逐个结果与训练段指标、不合法的表达式和错误、重复），最后一行 `end`（停止原因、累计 token 与费用）。

**转写记录和挖出来的因子都是私有研究输出**：路径落在仓库目录里（`data/` 除外）时直接报错，不要提交进仓库，文档和例子只用合成行情。

## 测试与变异检查

`test/factor/test_mining_llm_parse.py`（解析：合法写法、未知算子与字段、参数不在取值里、超深、节点过多、根有量纲、截面参数有量纲、数值常量、垃圾文本与 Python 代码、回写文本往返）、`test_mining_llm_client.py`（请求形状、429 / 5xx / 传输错误重试后放弃、其他 4xx 不重试、密钥不出现在报错里、缺密钥的提示、整次请求超时、用量与费用）、`test_mining_llm.py`（坏 JSON 反馈不崩溃、不合法的不评估并反馈、跨轮去重、两种预算与连续无效轮的停止、接口报错、历史与笔记与建议、白名单、转写记录位置与无密钥、`describe` 随提示词变、合成行情端到端找到埋入的信号且台账试验数等于评估的候选数、信息隔离、命令行、GPU 粗筛）。

变异检查（2026-11-14，每个变异单独打到代码上跑上面三个文件）：

| 变异 | 抓到它的测试 |
|---|---|
| 无改动（对照） | 全部通过 |
| `train_only` 原样返回（白名单失效） | `test_only_whitelisted_training_fields_reach_the_prompt` |
| `_train_feedback` 读验证段 | `test_mine_feedback_hook_reads_only_training_segment`、`test_prompts_see_only_the_training_segment` |
| 解析后不按空间校验（评估前校验失效） | `test_invalid_expressions_are_not_evaluated_and_fed_back` 等 |
| `_over_budget` 永远不停 | `test_budget_usd_stops_before_next_round`、`test_budget_tokens_stops` |
| 去掉跨轮去重 | `test_dedup_within_and_across_rounds` |
| 连续无效轮不计数 | `test_consecutive_rounds_without_valid_candidates_stop` |

## 经 OpenRouter 调用

```bash
python -m ashare.factor.mining.llm --base-url https://openrouter.ai/api/v1 \
  --key-env OPENROUTER_API_KEY --model deepseek/deepseek-v4.1-flash --rounds 20
```

OpenRouter 在 `usage.cost` 里给出实际扣费，有它时预算按它算（转写记录 `cost_source: reported`），并记下实际转到的服务商（`provider`）。DeepSeek Flash 默认思考：旧的 4096 输出上限会被思考占满、回答为空（2026-11-14 首次实跑三轮全空），现在默认 32768；如果仍然被截断成空回答，循环会立即停下（停止原因 `truncated`）并说明思考用了多少。
