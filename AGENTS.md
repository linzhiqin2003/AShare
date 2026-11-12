# AShare — 给实现者的说明

先读 `CLAUDE.md` 和 `docs/DESIGN.md`。

- 只改本项目目录里的文件；不提交、不推送；不装系统级软件。
- `packages/` 下的包（`panelbt`、`paperbroker`、`panelops`、`portopt`）不得 import `ashare`，也不互相 import，运行依赖只有 numpy。各包的 Rust 库名必须唯一（`panelbt_ext`、`paperbroker_native`、`panelops_ext`、`portopt_ext`），Python 模块名仍是 `<包>._core`。
- Tushare 的真实数据（整份接口响应、行情、面板）不进 git：放在 `data/` 下（`data/fixtures/` 给测试用），测试找不到时带原因跳过。测试和脚本不得写真实数据目录。
- 根 `.gitignore` 里忽略根目录的东西写成 `/name/`。
- 这三个包里的代码注释、文档字符串和 README 用英文（另附 `README.zh-CN.md`）；项目级文档（`docs/`）和 `src/ashare` 用中文。标识符一律用英文。
- 测试不许为了通过而放宽断言或跳过；发现设计有问题，写进当步的笔记并说明你的处理。
- 密钥只从环境变量读，任何输出和文件里都不出现密钥的值。
