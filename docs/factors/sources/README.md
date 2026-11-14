# 公开因子库原文

三套公开量价因子的公式原文和各自的函数、输入数据定义，是 `ashare.factor.formula` 翻译与核对的唯一依据。抄录方法、双来源比对结果和疑点清单见 `SOURCES_REPORT.md`（2026-11-13 抄录；入库时只把其中一处本机临时脚本路径改成了文件名，其余逐字未改）。

| 文件 | 内容 | 出处 | 许可情况 |
|---|---|---|---|
| `alpha101.jsonl` | WorldQuant Alpha101，101 条，每行一条 | Z. Kakushadze, "101 Formulaic Alphas", arXiv:1601.00991 v3, 附录 A.1；第二来源为 GitHub `yli188/WorldQuant_alpha101_code` 的代码注释，第三来源为聚宽 Alpha101 文档镜像 | arXiv 论文，版权归作者；arXiv 默认是非独占发布许可，不是开源许可。这里只为研究复现抄录公式，引用时注明出处 |
| `gtja191.jsonl` | 国泰君安 191，191 条 | 国泰君安证券《数量化专题之九十三：基于短周期价量特征的多因子选股体系》（2017-06-16）表 6；第二来源为聚宽「短周期价量特征 191 Alphas」文档镜像，部分条目另有一份公开 gist 的代码注释 | 券商研报，版权归国泰君安证券；公式来自公开发布的研报，这里只为研究复现抄录，引用时注明出处。聚宽文档声明对部分公式做过调整，不等同研报原文 |
| `alpha158.jsonl` | Qlib Alpha158，158 条（展开后的表达式） | microsoft/qlib main 分支 `qlib/contrib/data/loader.py`（`Alpha158DL.get_feature_config`），与 v0.8.6 `handler.py` 逐条一致 | MIT License, Copyright (c) Microsoft Corporation |
| `alpha158_defs.md` | Qlib 相关算子的源码原文（`ops.py`、`rolling.pyx`、`base.py`、`handler.py`，以及 v0.8.6 的 `Rank`） | microsoft/qlib main 与 v0.8.6 | MIT License, Copyright (c) Microsoft Corporation |
| `alpha158_loader_source.py.txt` | `qlib/contrib/data/loader.py` 全文（只读，不运行） | microsoft/qlib main | MIT License, Copyright (c) Microsoft Corporation |
| `SOURCES_REPORT.md` | 抄录与比对报告、不一致和疑点清单 | 本项目整理 | 随本仓库 |

Qlib 的 MIT 许可要求保留版权和许可声明：上面三个 Qlib 文件里的源码原文版权归 Microsoft Corporation，按 MIT License 使用（"Permission is hereby granted, free of charge, to any person obtaining a copy of this software ... THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND"，全文见 https://github.com/microsoft/qlib/blob/main/LICENSE）。

论文附录 A 的函数与输入定义（Alpha101）、研报附录 2 与聚宽「公用函数说明」（国君 191）的逐字抄录（`alpha101_defs.md`、`gtja191_defs.md`）不随公开仓库发布：那是论文、研报和聚宽文档的原文段落，版权归原作者；需要时按上表出处查原文。翻译与参考实现用到的口径都已写进 `../CONVENTIONS.md`。

这些文件只读：翻译、转换程序和参考实现都从这里取原文；任何纠错和取舍写在 `../CONVENTIONS.md`，不改原文。
