"""D1 集成：通过业务入口读写外置盘中的实际面板、缓存、台账与状态文件。"""

import numpy as np
import polars as pl

from .test_entries import pointer


def test_business_contents(roots):
    import ashare.data as ad
    import ashare.ml as ml
    from ashare.data.crypto import open as crypto_open
    from ashare.data.crypto.dataset import write_panel as crypto_write
    from ashare.factor.mining.ledger import open_ledger
    from ashare.paper.ops.context import Context, read_json, write_json
    from ashare.platform.config import DEFAULTS, platform_dir
    from ashare.platform.language import parse
    from ashare.platform.storage import Store
    from bench.platform.synthetic import write_panel

    root, external = roots
    target = external / "renamed-cn"
    target.mkdir()
    pointer(root, "cn", target)
    source = write_panel(root)
    np.testing.assert_array_equal(ad.open().panel("close")["close"], source["close"])
    assert ad.market_root() == external / "renamed-cn"
    assert ad.read_manifest()

    crypto = external / "renamed-crypto"
    frame = pl.DataFrame(
        dict(time=[1_700_000_000_000], instrument=["spot:TESTUSDT"], close=[42.5])
    )
    crypto_write(crypto, frame, profiles=[], events=[], data={}, notes={})
    pointer(root, "crypto", crypto)
    assert crypto_open().read()["close"].to_list() == [42.5]

    for name in ("ml", "mining", "paper", "platform"):
        target = external / f"renamed-{name}"
        target.mkdir()
        pointer(root, name, target)

    features = {"price": source["close"]}
    mask = np.ones(source.shape, dtype=bool)
    built = ml.build_store(features, source, mask)
    expected = built.column("price").copy()
    assert built.path.is_relative_to(external / "renamed-ml")
    reopened = ml.build_store(features, source, mask)
    np.testing.assert_array_equal(reopened.column("price"), expected)
    ledger = ml.Ledger()
    ledger.append(dict(event="trial", trial_id="ml-external"))
    assert ml.Ledger().records()[0]["trial_id"] == "ml-external"

    open_ledger().append(dict(event="trial", trial_id="mining-external"))
    assert open_ledger().records()[0]["trial_id"] == "mining-external"

    store = Store(platform_dir())
    store.begin(
        "tab", "external-factor", [(parse("rank(close)", factors={})[0], DEFAULTS)], {}
    )
    store.close()
    store = Store(platform_dir())
    assert store.library()["items"][0]["name"] == "external-factor"
    store.close()

    context = Context.create()
    write_json(context.ops / "state" / "test.json", {"outside": True})
    assert read_json(Context.create().ops / "state/test.json") == {"outside": True}
    for name in ("cn", "crypto", "ml", "mining", "paper", "platform"):
        assert [p.name for p in (root / name).iterdir()] == ["LOCATION.json"]
