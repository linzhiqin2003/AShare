"""核对工具的「已知例外」：负责人裁决过、候选与参考按约定就该不同的格子。

一条例外只对同时满足下面三条的格子生效，不会整条因子豁免：

1. 例外的条件（一个表达式，在候选的同一份数据上求值）在这一格为真（有限且非零）；
2. 这一格按正常规则不过关：掩码不同，或误差超过「一致」的阈值；
3. 候选在这一格恰好等于例外声明的值（逐位相等）。

这样的格子从误差统计里拿掉，单独计数并列出样例；条件成立但候选值不对的格子仍然算
不一致。具体的例外由因子包登记在 ``FactorSet.exceptions`` 里。
"""

from dataclasses import dataclass

from ..expression import Expr


@dataclass(frozen=True)
class KnownException:
    ref: str  # 因子包约定文件里的编号
    condition: Expr  # 为真（有限且非零）的格子才可能是例外
    candidate: float  # 例外格里候选必须给出的值
    description: str
