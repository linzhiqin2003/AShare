"""合成数据页面截图与交互验收；服务和浏览器退出时一并关闭。"""

import argparse
import html
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from bench.platform.http_perf import seed
from bench.platform.synthetic import panel, write_indexes, write_panel


def free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


REFERENCES = {
    "research_empty": "p1_research_light",
    "research_running": "p1_research_light",
    "research_results": "p1_research_light",
    "research_relative": "p1_research_light",
    "research_settings": "p1_research_settings_light",
    "research_benchmarks": "p1_research_settings_light",
    "research_completion": "p7_backtest_light",
    "research_window_half": "p1_research_light",
    "research": "p1_research_light",
    "library_flat": "p5_library_flat_light",
    "library_expression": "p5_library_light",
    "library_config_filter": "p5_library_byconfig_light",
    "library_drawer": "p5_library_flat_light",
}


def write_gallery(output, names):
    """每张完成即写入索引；中途失败仍可对照已完成的截图。设计稿只引用，不复制。"""
    (output / "screens.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2)
    )
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>第四轮视觉对照</title><style>
body{margin:16px;font:14px sans-serif;background:#f1f2f4;color:#222}
section{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:32px}
figure{margin:0;min-width:0}img{width:100%;height:auto;background:white}
figcaption{margin:8px 0;color:#59616b}h2{font-size:16px}
</style><h1>第四轮视觉对照</h1>
<p>左侧设计稿，右侧合成数据实现。空、运行中、基准名单及相对基准模式使用同页设计稿作布局参考；补全只对照候选弹窗。点击图片查看原尺寸。</p>"""
    for name in names:
        state = re.sub(r"_(1366|1920)$", "", name)
        reference = (
            Path(__file__).resolve().parents[2]
            / ".cache/review-r1"
            / f"{REFERENCES[state]}.png"
        )
        relative = html.escape(os.path.relpath(reference, output), quote=True)
        actual = html.escape(name, quote=True)
        document += f"<h2>{actual}</h2><section><figure><figcaption>设计稿</figcaption><a href='{relative}'><img alt='设计稿' src='{relative}'></a></figure><figure><figcaption>当前实现</figcaption><a href='{actual}.png'><img alt='当前实现' src='{actual}.png'></a></figure></section>"
    (output / "comparisons.html").write_text(document + "</html>")


def capture(page, output):
    """页面必须真实完成交互；不伪造后端结果、进度或页面 DOM。"""
    from playwright.sync_api import expect

    names, errors, interactions = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_default_timeout(15000)

    def settle():
        page.evaluate("document.fonts.ready")
        page.evaluate(
            "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
        )

    def shot(name):
        settle()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), name
        assert not errors, errors
        # 用实际 SVG 文字边界检查上下图轴标签的间隔，而不只检查 CSS 声明。
        for chart in page.locator(".result-chart:visible").all():
            main = chart.locator(".main-tick text")
            draw = chart.locator(".draw-tick text")
            texts = main.all_text_contents()
            assert len(texts) == len(set(texts)), (name, "收益轴重复标签", texts)
            if "compact" in chart.get_attribute("class").split():
                assert not draw.count(), (name, "抽屉回撤轴应隐藏")
            if main.count() and draw.count():
                bottoms = [
                    box["y"] + box["height"]
                    for box in (label.bounding_box() for label in main.all())
                ]
                top = min(label.bounding_box()["y"] for label in draw.all())
                assert top - max(bottoms) >= 8, (
                    name,
                    "收益与回撤轴标签间距",
                    top - max(bottoms),
                )
        # 使用浏览器测得的实际字框，守住窄屏与抽屉打开后的标签避让。
        for scatter in page.locator(".scatter-panel:visible svg").all():
            assert scatter.evaluate("""svg => {
              const labels = [...svg.querySelectorAll('.placed-name')].map(e => e.getBBox());
              const points = [...svg.querySelectorAll('circle')].map(e => e.getBBox());
              const overlaps = (a,b) => a.x < b.x+b.width && a.x+a.width > b.x && a.y < b.y+b.height && a.y+a.height > b.y;
              return labels.every((a,i) => a.x >= 40 && a.x+a.width <= 428 && a.y >= 22 && a.y+a.height <= 207 && points.every(b => !overlaps(a,b)) && labels.slice(i+1).every(b => !overlaps(a,b)));
            }"""), (name, "散点标签压点、越界或互相重叠")
        for label in page.locator(".checks:visible .check-label").all():
            assert label.evaluate(
                "el => getComputedStyle(el).whiteSpace === 'normal' && getComputedStyle(el).textOverflow !== 'ellipsis' && el.scrollWidth <= el.clientWidth + 1"
            ), (name, label.inner_text())
        page.screenshot(path=str(output / f"{name}.png"), animations="disabled")
        names.append(name)
        write_gallery(output, names)

    def nav(name):
        page.get_by_role("navigation").get_by_role(
            "button", name=name, exact=True
        ).click()

    def relative_chart(name, mode):
        page.get_by_label("收益累计方式", exact=True).select_option(mode)
        choices = page.get_by_label("超额收益曲线", exact=True)
        expect(choices).to_be_visible()
        expect(
            page.locator('.chart-panel .return-line[data-kind="long_short"]')
        ).to_have_count(4)
        expect(
            page.locator('.chart-panel .return-line[data-kind="long"]')
        ).to_have_count(4)
        expect(
            page.locator('.chart-panel .return-line[data-kind="benchmark"]')
        ).to_have_count(0)
        expect(page.locator(".benchmark-legend")).to_have_count(0)
        choices.select_option("long")
        expect(
            page.locator('.chart-panel .return-line[data-kind="long_short"]')
        ).to_have_count(0)
        expect(
            page.locator('.chart-panel .return-line[data-kind="long"]')
        ).to_have_count(4)
        expect(page.locator(".chart-panel svg")).to_contain_text("超额回撤 · 第 1 组")
        choices.select_option("both")
        expect(page.locator(".chart-panel .return-line")).to_have_count(8)
        svg = page.locator(".chart-panel svg").bounding_box()
        page.mouse.move(svg["x"] + svg["width"] * 0.45, svg["y"] + 45)
        expect(page.locator(".hover-tip")).to_contain_text("超额")
        page.mouse.move(20, 20)
        shot(name)
        page.get_by_label("收益累计方式", exact=True).select_option("sum")
        expect(
            page.locator('.chart-panel .return-line[data-kind="benchmark"]')
        ).to_have_count(1)
        expect(
            page.locator('.chart-panel .return-line[data-kind="long"]')
        ).to_have_count(0)

    def open_group_menu():
        page.get_by_role("button", name="视图", exact=True).click()
        anchor = page.locator(".group-menu-anchor").first
        # 保留 hover → click 的真实事件顺序，防止旧页面的点击关闭问题再次出现。
        anchor.hover()
        anchor.locator("button").first.click()
        expect(anchor.locator(".submenu")).to_be_visible()
        return anchor

    def group(name):
        anchor = open_group_menu()
        choice = anchor.locator(".submenu").get_by_role("button", name=name, exact=True)
        if anchor.locator(".push").first.inner_text() == name:
            choice.click()
        else:
            with page.expect_response(
                lambda response: (
                    "/api/library?" in response.url and response.status == 200
                )
            ):
                choice.click()
        expect(page.locator(".library-scroll")).to_have_attribute("aria-busy", "false")
        expect(page.locator(".view-menu")).to_have_count(0)

    def open_detail():
        expect(page.locator(".entry-row").first).to_be_visible()
        page.locator(".entry-row").first.click()
        expect(page.locator(".detail-metrics td.number").first).not_to_have_text("—")
        expect(page.locator(".detail-drawer .result-chart path").first).to_be_visible()

    def drag_boundary(key, dx=0, dy=0, restore=None):
        root = page.locator(f'[data-split="{key}"]')
        handle = root.locator(":scope > .split-boundary")
        horizontal = "horizontal" in root.get_attribute("class").split()
        dimension = "width" if horizontal else "height"
        coordinate = "x" if horizontal else "y"

        def sizes():
            settle()
            first = (
                root.locator(":scope > .split-child").nth(0).bounding_box()[dimension]
            )
            second = (
                root.locator(":scope > .split-child").nth(1).bounding_box()[dimension]
            )
            total = root.bounding_box()[dimension] - handle.bounding_box()[dimension]
            min_first, min_second = [
                float(root.get_attribute(f"data-min-{side}"))
                for side in ("first", "second")
            ]
            max_first, max_second = [
                float(root.get_attribute(f"data-max-{side}"))
                for side in ("first", "second")
            ]
            scale = min(1, total / (min_first + min_second))
            lower, upper = (
                max(min_first * scale, total - max_second),
                min(max_first, total - min_second * scale),
            )
            assert lower - 1 <= first <= upper + 1, (
                key,
                "first bounds",
                first,
                lower,
                upper,
            )
            assert min_second * scale - 1 <= second <= max_second + 1, (
                key,
                "second bounds",
                second,
            )
            assert abs(first + second - total) < 1, (
                key,
                "overflow",
                first,
                second,
                total,
            )
            return {
                "first": first,
                "second": second,
                "total": total,
                "min": lower,
                "max": upper,
            }

        def move_to(position):
            box = handle.bounding_box()
            center = {
                "x": box["x"] + box["width"] / 2,
                "y": box["y"] + box["height"] / 2,
            }
            page.mouse.move(center["x"], center["y"])
            page.mouse.down()
            center[coordinate] = position
            page.mouse.move(center["x"], center["y"], steps=12)
            page.mouse.up()
            return sizes()

        initial = sizes()
        box = handle.bounding_box()
        changed = move_to(
            box[coordinate] + box[dimension] / 2 + (dx if horizontal else dy)
        )
        assert abs(changed["first"] - initial["first"]) > 10, (key, "first unchanged")
        assert abs(changed["second"] - initial["second"]) > 10, (
            key,
            "second unchanged",
        )
        saved = page.evaluate("key => localStorage.getItem('ashare.split.' + key)", key)
        assert saved is not None
        page.reload(wait_until="domcontentloaded")
        if restore:
            restore()
        else:
            expect(page.locator(".parameters tbody tr")).to_have_count(4)
        expect(handle).to_be_visible()
        restored = sizes()
        assert (
            page.evaluate("key => localStorage.getItem('ashare.split.' + key)", key)
            == saved
        )
        for side in ("first", "second"):
            assert abs(restored[side] - changed[side]) <= 1, (
                key,
                "reload",
                changed,
                restored,
            )

        edge = root.bounding_box()
        minimum = move_to(edge[coordinate] + 0.5)
        maximum = move_to(edge[coordinate] + edge[dimension] - 0.5)
        assert abs(minimum["first"] - minimum["min"]) <= 1, (key, "minimum", minimum)
        assert abs(maximum["first"] - maximum["max"]) <= 1, (key, "maximum", maximum)
        handle.dblclick()
        reset = sizes()
        ratio = float(root.get_attribute("data-initial"))
        expected = max(reset["min"], min(reset["max"], reset["total"] * ratio))
        assert abs(reset["first"] - expected) <= 1, (key, "reset", reset, expected)
        assert (
            abs(
                float(
                    page.evaluate(
                        "key => localStorage.getItem('ashare.split.' + key)", key
                    )
                )
                - ratio
            )
            < 1e-8
        )
        interactions.append(
            {
                "split": key,
                "initial": initial,
                "dragged": changed,
                "reloaded": restored,
                "minimum": minimum,
                "maximum": maximum,
                "reset": reset,
            }
        )
        (output / "interactions.json").write_text(
            json.dumps(interactions, ensure_ascii=False, indent=2)
        )

    page.goto(page.base_url + "/#/research", wait_until="domcontentloaded")
    expect(page.get_by_role("button", name=re.compile("运行 4 组"))).to_be_enabled()
    shot("research_empty")
    page.get_by_role("button", name=re.compile("运行 4 组")).click()
    expect(page.get_by_role("button", name="取消", exact=True)).to_be_visible()
    shot("research_running")
    expect(page.get_by_role("button", name=re.compile("运行 4 组"))).to_be_visible(
        timeout=180000
    )
    expect(page.locator(".parameters tbody tr")).to_have_count(4)
    expect(page.locator(".result-chart path").first).to_be_visible()
    expect(page.locator(".benchmark-legend")).to_contain_text("上证指数")
    expect(page.locator(".parameters th").filter(has_text="年化超额")).to_have_count(1)
    expect(page.locator(".parameters th").filter(has_text="信息比率")).to_have_count(1)
    expect(page.locator(".checks th")).to_have_count(4)
    shot("research_results")
    relative_chart("research_relative", "relative-sum")
    original_tabs = page.request.get(page.base_url + "/api/tabs").json()
    page.reload(wait_until="domcontentloaded")
    expect(page.locator(".parameters tbody tr")).to_have_count(4)
    assert page.request.get(page.base_url + "/api/tabs").json() == original_tabs
    page.get_by_role("button", name="设置", exact=True).click()
    expect(page.locator(".settings-pop")).to_be_visible()
    run_box = page.get_by_role("button", name=re.compile("运行 4 组")).bounding_box()
    settings_box = page.locator(".settings-pop").bounding_box()
    assert settings_box["y"] + settings_box["height"] <= run_box["y"]
    shot("research_settings")
    page.get_by_role("button", name="选择基准", exact=True).click()
    expect(page.get_by_role("option").first).to_be_visible()
    assert "中证" in page.locator(".index-options").inner_text()
    shot("research_benchmarks")
    page.keyboard.press("Escape")
    expect(page.locator(".index-menu")).to_have_count(0)
    page.keyboard.press("Escape")
    expect(page.locator(".settings-pop")).to_have_count(0)
    editor = page.locator(".editor-panel .cm-content")
    original_source = editor.inner_text()
    editor.click()
    page.keyboard.press("ControlOrMeta+a")
    page.keyboard.insert_text(original_source + "\nrank(lo")
    expect(page.locator(".cm-tooltip-autocomplete")).to_be_visible()
    shot("research_completion")
    page.keyboard.press("Escape")
    page.keyboard.press("Meta+z" if sys.platform == "darwin" else "Control+z")
    page.keyboard.press("ArrowRight")
    slider = page.get_by_role("slider", name="时间窗起点")
    box = slider.bounding_box()
    brush = page.locator(".brush-body").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(
        brush["x"] + brush["width"] * 0.5, box["y"] + box["height"] / 2, steps=20
    )
    page.mouse.up()
    assert int(slider.get_attribute("aria-valuenow")) > 0
    page.mouse.move(20, 20)
    shot("research_window_half")
    # 两个边缘和整块平移都必须工作；检查状态发生变化。
    right = page.get_by_role("slider", name="时间窗终点")
    previous = int(right.get_attribute("aria-valuenow"))
    right.focus()
    page.keyboard.press("ArrowLeft")
    assert int(right.get_attribute("aria-valuenow")) < previous
    body = page.locator(".brush-body").bounding_box()
    before = int(slider.get_attribute("aria-valuenow"))
    page.mouse.move(body["x"] + body["width"] / 2, body["y"] + 10)
    page.mouse.down()
    page.mouse.move(body["x"] + body["width"] / 2 - 40, body["y"] + 10, steps=8)
    page.mouse.up()
    assert int(slider.get_attribute("aria-valuenow")) < before
    drag_boundary("research-columns", dx=60)
    drag_boundary("research-rows", dy=-40)
    drag_boundary("research-tables", dx=-40)

    def library_states(suffix="", test_split=False):
        nav("因子库")
        expect(page.locator(".entry-row").first).to_be_visible()
        group("不分组")
        shot("library_flat" + suffix)
        group("表达式")
        expect(page.locator(".group-row").first).to_be_visible()
        anchor = open_group_menu()
        submenu = anchor.locator(".submenu").bounding_box()
        parent = anchor.bounding_box()
        assert submenu["x"] >= parent["x"] + parent["width"]
        shot("library_expression" + suffix)
        page.keyboard.press("Escape")
        group("配置")
        page.get_by_role("button", name="筛选", exact=True).click()
        with page.expect_response(
            lambda response: "/api/library?" in response.url and response.status == 200
        ):
            page.get_by_label("股票池 TOP800", exact=True).uncheck()
        expect(page.locator(".library-scroll")).to_have_attribute("aria-busy", "false")
        shot("library_config_filter" + suffix)
        with page.expect_response(
            lambda response: "/api/library?" in response.url and response.status == 200
        ):
            page.get_by_role("button", name="清除", exact=True).click()
        expect(page.locator(".library-scroll")).to_have_attribute("aria-busy", "false")
        page.keyboard.press("Escape")
        group("不分组")
        open_detail()
        shot("library_drawer" + suffix)
        drawer = page.locator(".detail-drawer").bounding_box()
        scatter = page.locator(".scatter-panel").bounding_box()
        assert scatter["x"] + scatter["width"] <= drawer["x"]
        if test_split:
            drag_boundary("library-drawer", dx=-50, restore=open_detail)
        page.get_by_role("button", name="关闭详情", exact=True).click()

    library_states(test_split=True)
    for width, height in ((1366, 768), (1920, 1080)):
        page.set_viewport_size({"width": width, "height": height})
        library_states(f"_{width}")
        nav("研究台")
        expect(page.locator(".parameters tbody tr")).to_have_count(4)
        shot(f"research_{width}")
        if width == 1366:
            relative_chart("research_relative_1366", "relative-compound")
    # 直接路由和前进后退。
    page.goto(page.base_url + "/#/library", wait_until="domcontentloaded")
    expect(page.locator(".library-page")).to_be_visible()
    nav("研究台")
    expect(page.locator(".editor-panel")).to_be_visible()
    page.go_back()
    expect(page.locator(".library-page")).to_be_visible()
    page.go_forward()
    expect(page.locator(".editor-panel")).to_be_visible()
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".cache/review-r4")
    parser.add_argument("--browser", help="可选 Chromium 可执行文件")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / ".cache"):
        parser.error("截图与合成数据仅允许写入本 worktree 的 .cache")
    output.mkdir(parents=True, exist_ok=True)
    data = output / "synthetic-data"
    if not data.exists():
        source = write_panel(data, panel(n=120, full=True))
        write_indexes(data, source)
    results = output / "synthetic-results"
    if not results.exists():
        seed(results, count=24)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = dict(
        os.environ,
        ASHARE_DATA_DIR=str(data),
        ASHARE_PLATFORM_DIR=str(results),
        GIT_OPTIONAL_LOCKS="0",
        MPLCONFIGDIR=str(root / ".cache/mpl"),
    )
    executable = args.browser
    if executable is None:
        installed = sorted(
            (Path.home() / "Library/Caches/ms-playwright").glob(
                "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"
            ),
            reverse=True,
        )
        if installed:
            executable = str(installed[0])
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "ashare.platform", "--port", str(port)],
            env=env,
            cwd=root,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("服务启动失败，检查 server.log")
                try:
                    if httpx.get(url + "/api/meta", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("服务未在 25 秒内就绪")
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=executable, timeout=20000
                )
                try:
                    context = browser.new_context(
                        viewport={"width": 1600, "height": 1000}, device_scale_factor=2
                    )
                    page = context.new_page()
                    page.base_url = url
                    try:
                        names = capture(page, output)
                    except Exception:
                        page.screenshot(path=str(output / "failure.png"), timeout=10000)
                        raise
                    print(f"已完成 {len(names)} 张截图与布局/路由验收")
                finally:
                    browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=3)


if __name__ == "__main__":
    main()
