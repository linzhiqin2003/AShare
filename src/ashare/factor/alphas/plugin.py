"""入口点 ``ashare.factors``：把三套因子挂到 ``ashare.factor.formula`` 的核对工具上。"""

from ashare.factor.formula import FactorSet

from . import exceptions, registries


def register(hub):
    hub.add(
        FactorSet(
            "alpha158",
            registries.alpha158_registry,
            reference="ashare.factor.alphas.reference.alpha158",
            chunk=400,  # 全是时序运算：参考实现按证券分块并行
            exceptions=exceptions.ALPHA158,
            description="Qlib Alpha158（158 条，由 Qlib 原文确定性转换）",
        )
    )
    hub.add(
        FactorSet(
            "alpha101",
            registries.alpha101_registry,
            reference="ashare.factor.alphas.reference.alpha101_ref",
            options=("argmax",),
            description="WorldQuant Alpha101（逐条翻译）",
        )
    )
    hub.add(
        FactorSet(
            "gtja191",
            registries.gtja191_registry,
            reference="ashare.factor.alphas.reference.gtja191_ref",
            description="国泰君安 191（逐条翻译，收 186 条）",
        )
    )
