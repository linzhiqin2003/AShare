"""因子挖掘：搜索空间、三种搜索策略、试验台账、多重检验折算与留出段。

只有机制，不带任何挖出来的因子；台账和结果默认写到数据根目录下不入库的 ``mining/``。

```python
from ashare.factor.mining import Random, Split, mine
from ashare.factor.mining.synthetic import market
panel = market(t=900, n=150, strength=0.003)            # 合成行情，埋了一个已知信号
split = Split.recent(panel.dates, holdout_years=1, valid_years=1)
result = mine(panel, split=split, strategy=Random(n=60, seed=1), root="/tmp/mining-demo")
print(result.text())
```
"""

from .ledger import default_root, open_ledger, record_trial, trial_id
from .search import SearchResult, Split, Truncated, mine
from .space import CATALOG, Space, Unit, canonical, dedup_key
from .strategies import Enumerate, Evolution, Random

__all__ = [
    "CATALOG",
    "Enumerate",
    "Evolution",
    "Random",
    "SearchResult",
    "Space",
    "Split",
    "Truncated",
    "Unit",
    "canonical",
    "dedup_key",
    "default_root",
    "mine",
    "open_ledger",
    "record_trial",
    "trial_id",
]
