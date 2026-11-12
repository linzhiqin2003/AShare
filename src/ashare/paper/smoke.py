"""开盘后执行：uv run python -m ashare.paper.smoke --directory data/paper-smoke。"""

import sys

from ashare.paper.__main__ import main

if __name__ == "__main__":
    main(["smoke", "--symbols", "600000.SH,000001.SZ,300750.SZ", *sys.argv[1:]])
