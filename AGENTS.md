# AShare — 给实现者的说明

先读 `CLAUDE.md` 和 `docs/DESIGN.md`。

- 只改本项目目录里的文件；不提交、不推送；不装系统级软件。
- `panelbt`、`paperbroker`、`panelops` 不得 import `ashare`，运行依赖只有 numpy。
- 这三个包里的代码注释、文档字符串和 README 用英文（另附 `README.zh-CN.md`）；项目级文档（`docs/`）和 `src/ashare` 用中文。标识符一律用英文。
- 测试不许为了通过而放宽断言或跳过；发现设计有问题，写进当步的笔记并说明你的处理。
- 密钥只从环境变量读，任何输出和文件里都不出现密钥的值。
