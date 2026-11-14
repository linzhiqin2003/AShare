"""经 VPN 等其他网卡开放平台时的访问控制（2026-11-14 加入）。

默认仍只认本机地址；设了访问口令后，非本机来的请求必须先登录，本机访问免登录。
"""

import pytest
from fastapi.testclient import TestClient

from ashare.platform.__main__ import resolve_access
from ashare.platform.access import COOKIE, FAIL_LIMIT, LOCK_SECONDS, Access
from ashare.platform.app import create_app
from bench.platform.synthetic import write_panel

TOKEN = "correct-horse-battery-staple"
VPN = "10.8.0.5:8765"
REMOTE = ("10.8.0.9", 50123)
LOOPBACK = ("127.0.0.1", 50124)


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def make_client(tmp_path):
    data = tmp_path / "data"
    write_panel(data)
    clients = []

    def make(access, client=REMOTE, host=VPN):
        app = create_app(tmp_path / "platform", data_root=str(data), access=access)
        test_client = TestClient(
            app, base_url=f"http://{host}", client=client, follow_redirects=False
        )
        clients.append(test_client.__enter__())
        return test_client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def vpn_access(clock=None):
    return Access(hosts=frozenset({"10.8.0.5"}), token=TOKEN, clock=clock or Clock())


def login(client, token=TOKEN, origin=f"http://{VPN}"):
    headers = {"Origin": origin} if origin else {}
    return client.post("/login", data={"token": token}, headers=headers)


def test_default_still_rejects_other_hosts(make_client):
    client = make_client(None, client=LOOPBACK)
    assert client.get("/api/meta").status_code == 403
    assert (
        client.get("/api/meta", headers={"Host": "127.0.0.1:8765"}).status_code == 200
    )
    # 没设口令时登录页不存在，也不会出现登录拦截。
    assert client.get("/login", headers={"Host": "localhost"}).status_code != 401


def test_remote_requests_need_login_and_cookie_keeps_it(make_client):
    client = make_client(vpn_access())
    assert client.get("/api/meta").status_code == 401
    page = client.get("/")
    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert "访问口令" in client.get("/login").text

    wrong = login(client, "wrong-token-wrong-token")
    assert wrong.headers["location"] == "/login?e=1"
    assert "set-cookie" not in wrong.headers
    assert "口令不正确" in client.get("/login?e=1").text
    # 登录表单同样只接受同源提交。
    assert login(client, origin=None).status_code == 403
    assert login(client, origin="http://evil.example").status_code == 403

    ok = login(client)
    assert ok.status_code == 303 and ok.headers["location"] == "/"
    cookie = ok.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert TOKEN not in cookie  # cookie 存派生值，不存口令本身
    assert client.get("/api/meta").status_code == 200
    # 登录后写操作照样要同源。
    assert (
        client.post("/api/tabs", headers={"Origin": "http://evil.example"}).status_code
        == 403
    )
    assert (
        client.post("/api/tabs", headers={"Origin": f"http://{VPN}"}).status_code == 200
    )


def test_forged_cookie_and_unknown_host_are_rejected(make_client):
    client = make_client(vpn_access())
    client.cookies.set(COOKIE, "0" * 64)
    assert client.get("/api/meta").status_code == 401
    session = vpn_access().session
    client.cookies.set(COOKIE, session)
    assert client.get("/api/meta").status_code == 200
    assert client.get("/api/meta", headers={"Host": "evil.example"}).status_code == 403
    # 换口令后旧登录失效。
    other = Access(hosts=frozenset({"10.8.0.5"}), token="another-long-token-000")
    assert other.session != session


def test_loopback_owner_skips_login_only_on_loopback_names(make_client):
    owner = make_client(vpn_access(), client=LOOPBACK, host="127.0.0.1:8765")
    assert owner.get("/api/meta").status_code == 200
    assert owner.get("/api/meta", headers={"Host": "localhost:8765"}).status_code == 200
    # 本机连接但用 VPN 地址访问，按远程对待。
    assert owner.get("/api/meta", headers={"Host": VPN}).status_code == 401


def test_repeated_failures_lock_the_source_then_expire(make_client):
    clock = Clock()
    client = make_client(vpn_access(clock))
    for _ in range(FAIL_LIMIT):
        assert (
            login(client, "wrong-token-wrong-token").headers["location"].endswith("e=1")
        )
    locked = login(client)
    assert locked.headers["location"] == "/login?e=2"
    assert "set-cookie" not in locked.headers
    clock.now += LOCK_SECONDS + 1
    assert login(client).headers["location"] == "/"


def test_host_matching_is_exact():
    access = vpn_access()
    assert access.host_allowed("10.8.0.5:8765")
    assert access.host_allowed("LOCALHOST:8765")
    assert access.host_allowed("127.0.0.1")
    for host in ("10.8.0.50:8765", "10.8.0.5.evil:80", "10.8.0.5:x", "", "evil"):
        assert not access.host_allowed(host)
    with pytest.raises(ValueError):
        Access(token="short")


@pytest.mark.parametrize(
    ("host", "token", "message"),
    [
        ("10.8.0.5", None, "访问口令"),
        ("0.0.0.0", TOKEN, "0.0.0.0"),
        ("::", TOKEN, "IPv4"),
        ("vpn.example", TOKEN, "IPv4"),
        ("10.8.0.5", "short", "至少"),
    ],
)
def test_resolve_access_refuses_unsafe_setups(host, token, message):
    with pytest.raises(ValueError, match=message):
        resolve_access(host, [], token)


def test_resolve_access_binds_loopback_plus_one_address():
    binds, access = resolve_access(None, [], None)
    assert binds == ["127.0.0.1"] and access.token is None
    binds, access = resolve_access("10.8.0.5", ["vpn.example"], TOKEN)
    assert binds == ["127.0.0.1", "10.8.0.5"]
    assert access.host_allowed("vpn.example:8765") and access.token == TOKEN
