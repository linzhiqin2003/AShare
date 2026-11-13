"""平台是可选依赖（`uv sync --extra platform`）；没装 FastAPI 时跳过依赖它的测试，其余照常跑。"""

import importlib.util

collect_ignore = (
    [] if importlib.util.find_spec("fastapi") else ["test_api.py", "test_benchmarks.py"]
)
