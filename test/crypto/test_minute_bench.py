"""C7 验收脚本回归：缓存同步不能被当作首次下载的速度。"""

import pytest

from bench.crypto_minute_sample import estimate_download_seconds


@pytest.mark.parametrize("downloaded", [0, 50])
def test_cached_sync_cannot_extrapolate_network_throughput(downloaded):
    report = dict(seconds=1.0, downloaded_bytes=downloaded)
    assert estimate_download_seconds(report, 100, 1000) is None


def test_cold_sync_extrapolates_measured_bytes():
    report = dict(seconds=10.0, downloaded_bytes=105)
    assert estimate_download_seconds(report, 100, 1000) == 100.0
