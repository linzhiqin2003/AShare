"""访问控制：允许的 Host 和口令登录。

默认只听本机回环地址、不需要登录。监听其他网卡（例如 VPN 地址）时必须设访问口令：
非本机来的请求先登录，登录后靠 HttpOnly cookie 保持；本机回环来的请求不受影响。
"""

import hashlib
import hmac
import ipaddress
import logging
import time
from dataclasses import dataclass, field

LOG = logging.getLogger("ashare.platform")

LOOPBACK_HOSTS = ("localhost", "127.0.0.1")
COOKIE = "ashare_session"
COOKIE_AGE = 30 * 24 * 3600
MIN_TOKEN = 16
# 每个来源地址 10 分钟内最多失败 10 次，超出后锁 10 分钟。
FAIL_WINDOW, FAIL_LIMIT, LOCK_SECONDS = 600.0, 10, 600.0


def is_loopback(address):
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return address == "localhost"


@dataclass
class Access:
    hosts: frozenset = frozenset(LOOPBACK_HOSTS)
    token: str | None = None
    clock: object = time.monotonic
    _failures: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.hosts = frozenset(h.lower() for h in self.hosts) | set(LOOPBACK_HOSTS)
        if self.token is not None and len(self.token) < MIN_TOKEN:
            raise ValueError(f"访问口令至少 {MIN_TOKEN} 个字符")

    @property
    def session(self):
        # cookie 里存口令派生值，不存口令本身；换口令即让旧登录全部失效。
        return hmac.new(
            self.token.encode(), b"ashare-platform-session", hashlib.sha256
        ).hexdigest()

    def host_allowed(self, host):
        name, _, port = host.rpartition(":") if ":" in host else (host, "", "")
        if port and not (port.isdigit() and len(port) <= 5):
            return False
        return name.lower() in self.hosts

    def needs_login(self, client, host):
        """本机回环连接用本机名访问时免登录；其余在设了口令时都要登录。"""
        if self.token is None:
            return False
        name = host.rpartition(":")[0] if ":" in host else host
        return not (client and is_loopback(client) and name.lower() in LOOPBACK_HOSTS)

    def logged_in(self, cookie_header):
        for part in cookie_header.split(";"):
            key, _, value = part.strip().partition("=")
            if key == COOKIE and hmac.compare_digest(value, self.session):
                return True
        return False

    def locked(self, client):
        now = self.clock()
        times = [t for t in self._failures.get(client, ()) if now - t < FAIL_WINDOW]
        self._failures[client] = times
        return len(times) >= FAIL_LIMIT and now - times[-1] < LOCK_SECONDS

    def check(self, client, given):
        if hmac.compare_digest(given.encode(), self.token.encode()):
            self._failures.pop(client, None)
            LOG.info("登录成功 client=%s", client)
            return True
        self._failures.setdefault(client, []).append(self.clock())
        LOG.warning("登录失败 client=%s", client)
        return False

    def cookie(self):
        return (
            f"{COOKIE}={self.session}; Path=/; Max-Age={COOKIE_AGE}; "
            "HttpOnly; SameSite=Strict"
        )


LOGIN_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AShare</title>
<style>
:root{--ink:#18181b;--ink3:#71717a;--line:#e4e4e7;--red:#c2410c}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#fafafa;
font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif;color:var(--ink)}
form{width:min(320px,calc(100vw - 32px));display:grid;gap:12px}
b{font-size:15px;font-weight:600;margin-bottom:4px}
input{height:36px;border:1px solid var(--line);border-radius:4px;padding:0 10px;font:inherit;background:#fff}
input:focus{outline:none;border-color:var(--ink)}
button{height:36px;border:0;border-radius:4px;background:var(--ink);color:#fff;font:inherit;cursor:pointer}
p{margin:0;font-size:12px;color:var(--red);min-height:18px}
</style></head><body>
<form method="post" action="/login">
<b>AShare</b>
<input type="password" name="token" aria-label="访问口令" placeholder="访问口令" autocomplete="current-password" autofocus required>
<button type="submit">进入</button>
<p>__MESSAGE__</p>
</form></body></html>"""

MESSAGES = {"": "", "1": "口令不正确", "2": "尝试次数过多，请稍后再试"}


def login_page(code):
    return LOGIN_PAGE.replace("__MESSAGE__", MESSAGES.get(code, "")).encode()
