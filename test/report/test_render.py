"""render_html：可解析、内嵌数据能还原、不引用任何外部资源、模板随包分发。"""

import json
import re
from html.parser import HTMLParser
from importlib import resources

import pytest

from ashare.report import Benchmark, render_html, to_html
from ashare.report.render import EXTRA_KEYS, expand
from test.report.helpers import build, make_case


class Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.attrs, self.scripts, self._in = [], [], {}, None

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend(attrs)
        if tag == "script":
            self._in = dict(attrs).get("id") or f"script{len(self.scripts)}"
            self.scripts[self._in] = ""

    def handle_endtag(self, tag):
        if tag == "script":
            self._in = None

    def handle_data(self, data):
        if self._in:
            self.scripts[self._in] += data


@pytest.fixture(scope="module")
def ts():
    return build(make_case())


def parse(html):
    p = Collect()
    p.feed(html)
    p.close()
    return p


def close(a, b, tol):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            close(x, y, tol) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


def test_embedded_data_round_trips(ts, tmp_path):
    path = render_html(ts, tmp_path / "r.html")
    html = path.read_text(encoding="utf-8")
    p = parse(html)
    data = expand(json.loads(p.scripts["data"]))
    # 补充内容（render.EXTRA_KEYS）这次不上页面、不嵌入；其余键照旧全部嵌入
    want = {k: v for k, v in ts.items() if k not in EXTRA_KEYS}
    want["rebalances"] = [
        {k: v for k, v in r.items() if k != "unfilled_detail"} for r in ts["rebalances"]
    ]
    assert close(want, data, 1e-5)
    assert data["summary"] == ts["summary"]  # 汇总指标原样嵌入，不取整


def test_no_external_resources(ts):
    html = to_html(ts)
    assert "http://" not in html and "https://" not in html
    p = parse(html)
    assert "link" not in p.tags and "iframe" not in p.tags and "img" not in p.tags
    for k, v in p.attrs:
        assert k != "src", (k, v)
    assert "@import" not in html and "url(" not in html.replace("url(#", "")


def test_page_has_no_version_or_tool_name(ts):
    html = to_html(ts)
    for word in ("panelbt", "ashare", "version", "版本"):
        assert word not in html.lower()


def test_script_injection_is_neutralised(tmp_path):
    t = build(make_case(T=80))
    t["meta"]["title"] = "</script><script>alert(1)</script>"
    t["meta"]["description"] = "<!-- x </script>"
    html = to_html(t)
    p = parse(html)
    assert set(p.scripts) == {"data", "script1"}
    assert (
        expand(json.loads(p.scripts["data"]))["meta"]["description"]
        == "<!-- x </script>"
    )
    assert "<title>&lt;/script&gt;" in html


def test_template_is_packaged():
    text = (
        resources.files("ashare.report")
        .joinpath("templates/tearsheet.html")
        .read_text("utf-8")
    )
    assert "__TEARSHEET_DATA__" in text and "@media print" in text


@pytest.mark.parametrize("variant", ["no_bench", "late_bench", "no_industry", "short"])
def test_variants_render(variant, tmp_path):
    if variant == "short":
        t = build(make_case(T=90, every=5))
    else:
        c = make_case(T=200)
        if variant == "no_bench":
            t = build(c, benchmark=False)
        elif variant == "late_bench":
            b = c["bench"].copy()
            b[:40] = float("nan")
            t = build(c, benchmark=Benchmark(b, "x", "price"))
        else:
            t = build(c, industry=False)
    html = render_html(t, tmp_path / f"{variant}.html").read_text("utf-8")
    data = json.loads(parse(html).scripts["data"])
    assert re.search(r"<title>[^<]+</title>", html)
    assert data["daily"]["date"] == t["daily"]["date"]
