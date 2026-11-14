"""常驻服务的网段跟随（2026-11-14 用户要求平台改成常驻服务：开机时 VPN 未连上、
VPN 重连换地址都不能要人去重启）。"""

import pytest

from ashare.platform.__main__ import RESTART_CODE, subnet_address, watch_subnet

VPN = "10.77.0.0/24"


def test_subnet_address_picks_the_address_on_that_subnet():
    nics = {"lo0": ["127.0.0.1"], "en0": ["192.168.1.5"], "utun9": ["10.77.0.2"]}
    assert subnet_address(VPN, nics) == "10.77.0.2"
    assert subnet_address(VPN, {"lo0": ["127.0.0.1"], "en0": ["10.78.0.2"]}) is None
    # 网段写成主机地址也接受；同网段多个地址时取最小的，结果稳定
    assert (
        subnet_address("10.77.0.9/24", {"a": ["10.77.0.9"], "b": ["10.77.0.3"]})
        == "10.77.0.3"
    )


def test_subnet_address_rejects_ipv6_subnet():
    with pytest.raises(ValueError):
        subnet_address("fd00::/64", {})


class Stop(BaseException):  # 监视器吞掉 Exception，停止信号必须绕开它
    pass


def run_watch(bound, readings):
    """按顺序喂网卡读数；返回退出码（没退出时为 None）。"""
    seq = iter(readings)
    codes = []

    def read(_subnet):
        value = next(seq, Stop)
        if value is Stop:
            raise Stop
        if isinstance(value, Exception):
            raise value
        return value

    def exit_(code):
        codes.append(code)

    with pytest.raises(Stop) if not _changes(bound, readings) else _noop():
        watch_subnet(VPN, bound, read=read, exit=exit_, sleep=lambda _s: None)
    return codes[0] if codes else None


def _changes(bound, readings):
    return any(not isinstance(r, Exception) and r != bound for r in readings)


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_watch_keeps_running_while_address_is_stable():
    assert run_watch("10.77.0.2", ["10.77.0.2", "10.77.0.2"]) is None


@pytest.mark.parametrize(
    "bound, readings",
    [
        ("10.77.0.2", ["10.77.0.2", None]),  # VPN 断开：地址消失
        (None, [None, "10.77.0.2"]),  # 开机后 VPN 才连上
        ("10.77.0.2", ["10.77.0.5"]),  # 重连换了地址
    ],
)
def test_watch_exits_for_restart_when_address_changes(bound, readings):
    assert run_watch(bound, readings) == RESTART_CODE


def test_watch_skips_a_failed_read_instead_of_exiting():
    assert run_watch("10.77.0.2", [OSError("busy"), "10.77.0.2"]) is None
