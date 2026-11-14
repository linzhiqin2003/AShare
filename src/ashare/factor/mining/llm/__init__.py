"""大模型研究循环：让模型提出假设、写成表达式，由 `mine` 的评估器在训练段上打分并反馈。

思路参考微软 RD-Agent（R&D-Agent-Quant，MIT）的研究循环，但不依赖它：模型只写本仓库的
表达式语言，提示词自己写。说明见 ``docs/LLM_MINING.md``。

```python
from ashare.factor.mining import Split, mine
from ashare.factor.mining.llm import ChatClient, LLMResearch
from ashare.factor.mining.synthetic import market
panel = market(t=900, n=150, strength=0.003)
split = Split.recent(panel.dates, holdout_years=1, valid_years=1)
strategy = LLMResearch(ChatClient(), rounds=10, batch=8, budget_usd=1)  # 密钥：DEEPSEEK_API_KEY
result = mine(panel, split=split, strategy=strategy)
print(result.text(), strategy.last_run)
```

命令行：``python -m ashare.factor.mining.llm --rounds 20 --batch 8 --budget-usd 2 [--synthetic]``。
"""

from .client import ChatClient, LLMError, Pricing, Reply, Usage
from .parse import ParseError, parse, render
from .prompt import TEMPLATE_VERSION, template_sha
from .strategy import LLMResearch

__all__ = [
    "TEMPLATE_VERSION",
    "ChatClient",
    "LLMError",
    "LLMResearch",
    "ParseError",
    "Pricing",
    "Reply",
    "Usage",
    "parse",
    "render",
    "template_sha",
]
