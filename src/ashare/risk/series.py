"""按日期前缀构建快照；缓存同时保存后续估计需要的状态。"""

import ctypes
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .cache import ModelCache, report_failure
from .data import FIELDS, adjusted_returns
from .engine import cross_section_batch
from .fundamental import (
    FundamentalConfig,
    cross_sectional_bias,
    estimate_covariances,
    exposures_at,
)
from .kernels import exp_weights
from .model import RiskModel
from .rolling import RollingDescriptors
from .specific import SpecificMoments
from .statistical import statistical_model
from .window import ResidualWindow

CACHE_SCHEMA = 3


class _Precomputed:
    """按日给出已经顺序算好的滚动描述子，供各线程并行使用。"""

    def __init__(self, values):
        self.values = values

    def at(self, day):
        return self.values[day]


MAX_DEFAULT_THREADS = 8
MAX_THREADS = 256


def default_threads():
    """ASHARE_RISK_THREADS，否则取性能核数（Apple 芯片的能效核不计）或 CPU 数，最多 8。

    上限的理由：暴露计算里有一部分持 GIL，P-007 实测超过性能核数后反而变慢；
    其中的 `np.linalg.lstsq` 走 BLAS/LAPACK，库自己可能再开线程（OpenBLAS、MKL 的
    OMP/OPENBLAS_NUM_THREADS），总线程数约为两者之积。服务器上要开更多线程时，
    同时把 BLAS 线程设成 1。环境变量为空串时按未设置处理。
    """
    value = os.environ.get("ASHARE_RISK_THREADS")
    if value:
        if (
            not (value.isascii() and value.isdigit())
            or not 1 <= int(value) <= MAX_THREADS
        ):
            raise ValueError(f"ASHARE_RISK_THREADS: 必须为 1–{MAX_THREADS} 的整数")
        return int(value)
    count = None
    try:
        libc = ctypes.CDLL(None)
        found = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(found))
        if (
            libc.sysctlbyname(
                b"hw.perflevel0.physicalcpu",
                ctypes.byref(found),
                ctypes.byref(size),
                None,
                0,
            )
            == 0
            and found.value > 0
        ):
            count = found.value
    except (OSError, AttributeError):
        pass
    return max(1, min(count or os.cpu_count() or 1, MAX_DEFAULT_THREADS))


class _ExposurePool:
    """暴露计算的线程池，按需创建。

    fork 出的子进程里继承来的线程池没有工作线程（提交的任务永远不会执行），它的锁也可能
    停在父进程的状态，所以子进程不碰它，另起一个；关闭时也只关本进程建的。
    """

    def __init__(self, threads):
        self.threads = threads
        self.pool = None
        self.pid = None

    def map(self, fn, items):
        if self.pool is None or self.pid != os.getpid():
            self.pool = ThreadPoolExecutor(
                self.threads, thread_name_prefix="risk-exposure"
            )
            self.pid = os.getpid()
        return list(self.pool.map(fn, items))

    def shutdown(self):
        pool, self.pool = self.pool, None
        if pool is not None and self.pid == os.getpid():
            pool.shutdown()


def _row_parts(source, external, day):
    """第 day 天进入前缀链的字节，顺序与旧实现的 `h.update` 完全一致。"""
    yield str(source.dates[day]).encode()
    for field in FIELDS:
        try:
            value = np.ascontiguousarray(source[field][day])
        except KeyError:
            continue
        yield field.encode() + value.dtype.str.encode() + value.tobytes()
    if external is not None:
        yield np.ascontiguousarray(external[day]).tobytes()


def _row_digests(source, external, days, threads):
    """逐日输入的独立摘要（不含前缀），按日期分段交给线程池；hashlib 释放 GIL。"""
    out = np.zeros((len(days), 32), np.uint8)

    def run(chunk):
        for i in chunk:
            h = hashlib.sha256()
            for part in _row_parts(source, external, days[i]):
                h.update(part)
            out[i] = np.frombuffer(h.digest(), np.uint8)

    step = 256
    chunks = [range(i, min(i + step, len(days))) for i in range(0, len(days), step)]
    if threads == 1 or len(chunks) < 2:
        for chunk in chunks:
            run(chunk)
    else:
        with ThreadPoolExecutor(threads, thread_name_prefix="risk-prefix") as pool:
            list(pool.map(run, chunks))
    return out


def _prefix_chain(source, external, end, cache, threads):
    """逐日前缀链摘要（与旧实现逐位相同，进入缓存键和元数据 `prefix_hash`）。

    链是顺序的：第 t 天的摘要 = SHA-256(第 t−1 天的摘要 ‖ 第 t 天的输入)。
    缓存目录里存一份索引：每天输入的独立摘要和链摘要。下次先并行重算每天的独立摘要，
    与索引一致的最长前缀直接沿用索引里的链摘要（输入逐日相同，链必然相同），
    从第一个不一致或新增的日期起才顺序重算链。历史被改写照样能发现。
    """
    days = list(range(end))
    rows = _row_digests(source, external, days, threads) if cache else None
    index = cache.load_index() if cache else None
    reuse = 0
    if index is not None:
        stored_rows, stored_prefixes = index
        limit = min(len(stored_rows), end)
        same = np.all(stored_rows[:limit] == rows[:limit], axis=1)
        reuse = limit if same.all() else int(np.argmin(same))
    prefixes = [bytes(stored_prefixes[i]) for i in range(reuse)] if reuse else []
    prefix = prefixes[-1] if prefixes else b""
    for day in range(reuse, end):
        h = hashlib.sha256(prefix)
        for part in _row_parts(source, external, day):
            h.update(part)
        prefix = h.digest()
        prefixes.append(prefix)
    # 索引只在变长或历史改了时重写；只看了较短一段且全部一致时保留原来更长的索引。
    if cache and (index is None or reuse < end or len(index[0]) < end) and end:
        chain = np.frombuffer(b"".join(prefixes), np.uint8).reshape(-1, 32)
        cache.save_index(rows, chain)
    return prefixes


def _plan(cache, keys, start, end, history):
    """决定从哪天开始逐日处理，并预读需要的缓存状态。

    第一个未命中的日期 d 之前的日子不必重算；它们的状态只在 d 的估计窗口
    （最近 history 天）里用到，所以只预读 [d − history, d) 这一段，不再逐个打开全部历史文件。
    交给调用方的日子永远是 [start, end)：start 早于窗口时，[start, d − history) 这些日子
    在主循环里照常带模型读出、交出（这一段的读取是必须付的）。全部命中时只读 [start, end)。
    预读中途发现快照损坏或被换掉，就把 d 提前到那一天再来一遍。
    返回 (主循环起点, d, 预读的状态)。
    """
    if not cache:
        return 0, 0, {}
    miss = next((d for d in range(end) if cache.entries.get(d) != keys[d]), end)
    while True:
        window = max(0, miss - history) if miss < end else start
        first = min(start, window)
        preloaded = {}
        # start 之后的日子在主循环里带模型读（逐个交给调用方，不一次全读进内存）。
        for day in range(first, min(miss, start)):
            # 第一个未命中日的前一天要完整状态（回归和偏差统计用）；更早的只进历史窗口。
            fields = None if day == miss - 1 else ("f", "residual", "bias")
            loaded = cache.load(keys[day], day=day, with_model=False, fields=fields)
            if loaded is None:
                miss = day
                break
            preloaded[day] = loaded
        else:
            return first, miss, preloaded


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def model_series(
    source,
    *,
    config=None,
    cache=None,
    start=0,
    end=None,
    external=None,
    external_names=(),
    kind="fundamental",
    statistical_options=None,
    batch_size=64,
    threads=None,
):
    """逐日迭代。start 前的缓存状态仍参与估计；追加不会重算已缓存快照。

    version 表示数据修订版本；新增日期可以保持该版本。每个键还包含输入
    前缀摘要，历史改写即便未更新 version 也不会命中旧缓存。

    交出的日子是 [start, end)，与缓存命中多少无关。

    threads：一批日期的风格暴露并行计算、以及逐日输入摘要的线程数（结果与 1 个线程逐位相同）；
    None 取环境变量 ASHARE_RISK_THREADS，没有则取性能核数（最多 8，见 `default_threads`）。
    threads > 1 时 source 会被多个线程同时读取（`source[field][day]`），自定义的惰性加载
    数据源须能并发读；不能时传 threads=1。

    缓存写入在后台线程里进行；生成器结束（耗尽或显式 close()）前全部落盘，写失败在那里抛出。
    生成器没耗尽就被丢掉（break 后被回收）时异常无法传给调用方：错误写到 stderr，
    缓存对象保持失败状态，之后对它的 save/load/flush 都会抛出。

    多进程：子进程要跑本函数请用 spawn，或在父进程做任何计算之前 fork。fork 后线程池和
    缓存写线程会在子进程里重建，但 `portopt` 用的 rayon 全局线程池不能跨 fork 使用。
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size: 必须为正整数")
    threads = default_threads() if threads is None else threads
    if (
        not isinstance(threads, int)
        or isinstance(threads, bool)
        or not 1 <= threads <= MAX_THREADS
    ):
        raise ValueError(f"threads: 必须为 1–{MAX_THREADS} 的整数")
    c = FundamentalConfig() if config is None else config
    end = len(source.dates) if end is None else end
    if not 0 <= start <= end <= len(source.dates) or kind not in (
        "fundamental",
        "statistical",
    ):
        raise ValueError("日期范围或模型类型错误")
    if external is not None and np.asarray(external).shape[:2] != (
        len(source.dates),
        len(source.symbols),
    ):
        raise ValueError("external: 需要 T×N×K 面板")
    cache = ModelCache(cache) if isinstance(cache, (str, Path)) else cache
    params = _digest(
        {
            "schema": CACHE_SCHEMA,
            "version": source.version,
            "symbols": source.symbols,
            "config": asdict(c),
            "kind": kind,
            "statistical": statistical_options,
            "external_names": external_names,
        }
    )
    if cache:
        cache.prepare(params, source.symbols)
    returns = (
        adjusted_returns(source, end - 1) if end else np.empty((0, len(source.symbols)))
    )
    try:
        returns -= source["risk_free"][:end]
    except KeyError:
        pass
    returns[np.asarray(source["is_suspended"][:end], bool)] = np.nan
    # 前缀上市日数只扫描一次；每个元素只依赖该日期及更早的上市状态。
    listing_ages = np.cumsum(
        np.asarray(source["is_listed"][:end], bool), axis=0, dtype=np.int32
    )
    rolling = RollingDescriptors(source, returns, c, listing_ages)

    def exposure_for(day, descriptors=None):
        return exposures_at(
            source,
            returns,
            day,
            c,
            external,
            external_names,
            listing_ages=listing_ages,
            rolling=rolling if descriptors is None else descriptors,
        )

    pool = _ExposurePool(threads)

    def exposure_batch(days):
        """滚动描述子按日顺序递推（有状态），其余各天独立，交给线程池。"""
        descriptors = _Precomputed({day: rolling.at(day) for day in days})
        if threads == 1 or len(days) < 2:
            return [exposure_for(day, descriptors) for day in days]
        return pool.map(lambda day: exposure_for(day, descriptors), days)

    def statistical_for(day):
        options = dict(statistical_options or {})
        options.setdefault("components", min(45, len(source.symbols) - 1))
        return statistical_model(
            returns[max(0, day - c.history + 1) : day + 1],
            source.symbols,
            date=str(source.dates[day]),
            **options,
        )

    def cached_exposure(day):
        if kind == "statistical" and day >= c.min_descriptor_days:
            return statistical_for(day).B
        return exposure_for(day)[0]

    closing = failed = False
    try:
        prefixes = _prefix_chain(source, external, end, cache, threads)
        yield from _iterate(
            prefixes,
            source,
            c,
            cache,
            start,
            end,
            external,
            kind,
            params,
            returns,
            rolling,
            exposure_batch,
            statistical_for,
            cached_exposure,
            batch_size,
        )
    except GeneratorExit:
        closing = True
        raise
    except BaseException:
        failed = True
        raise
    finally:
        pool.shutdown()
        # 后台写入在生成器结束（含提前关闭）前落盘；写失败在这里抛出。
        if cache:
            try:
                cache.flush()
            except RuntimeError as error:
                if not failed:
                    if closing:
                        # 被回收关闭时异常只会被解释器忽略：先写到 stderr。
                        report_failure(cache, error, "生成器关闭时")
                    raise


def _iterate(
    prefixes,
    source,
    c,
    cache,
    start,
    end,
    external,
    kind,
    params,
    returns,
    rolling,
    exposure_batch,
    statistical_for,
    cached_exposure,
    batch_size,
):
    prepared, regressed = {}, {}
    history_f, history_bias = [], []
    # 残差历史放在预分配的双份环形缓冲里，窗口是连续视图（不再每天拼 48 MB 新数组）。
    residual_window = ResidualWindow(len(source.symbols), c.history)
    # 特质方差分块累加；从头算模式下为 None，与旧实现同一路径。
    moments = (
        SpecificMoments(len(source.symbols), c, residual_window)
        if c.specific_incremental
        else None
    )
    previous = None
    previous_key = None
    keys = [hashlib.sha256(params.encode() + p).hexdigest() for p in prefixes]
    first, miss, preloaded = _plan(cache, keys, start, end, c.history)
    for day in range(first, end):
        prefix = prefixes[day]
        key = keys[day]
        if day in preloaded:
            # 第一个未缓存日之前、窗口之内的日子：状态已在计划阶段读过。
            cached = preloaded.pop(day)
        else:
            cached = (
                cache.load(
                    key,
                    day=day,
                    with_model=day >= start,
                    exposure_loader=lambda day=day: cached_exposure(day),
                )
                if cache
                else None
            )
        if cached is None and day < miss:
            # 预计会命中的快照读失败（损坏或被换掉）：这一天要重算，但它的估计窗口
            # 可能早于已读的范围，前一天也可能只预读了部分字段。已经交出的日子不再交出，
            # 从这一天起重新计划。
            yield from _iterate(
                prefixes,
                source,
                c,
                cache,
                max(start, day),
                end,
                external,
                kind,
                params,
                returns,
                rolling,
                exposure_batch,
                statistical_for,
                cached_exposure,
                batch_size,
            )
            return
        if cached is not None:
            model, state = cached
        else:
            if day not in prepared:
                if cache and previous_key is not None:
                    checkpoint = cache.checkpoint(previous_key)
                    if checkpoint is not None and int(checkpoint["day"]) == day - 1:
                        rolling.restore(checkpoint)
                stop = min(end, day + batch_size)
                days = list(range(day, stop))
                prepared.update(zip(days, exposure_batch(days), strict=True))
                if kind == "fundamental":
                    days, designs, caps, estimates = [], [], [], []
                    for future in range(max(1, day), stop):
                        if future == day:
                            prev_model, prev_state = previous
                            if prev_model is None:
                                prev_model, _ = cache.load(
                                    previous_key,
                                    exposure_loader=lambda day=day: cached_exposure(
                                        day - 1
                                    ),
                                )
                            bx, cx, ex = (
                                prev_model.B,
                                prev_state["cap"],
                                prev_state["estimation"],
                            )
                        else:
                            bx, _, ex, cx = prepared[future - 1]
                        eligible = ex.astype(bool) & np.isfinite(returns[future])
                        if eligible.sum() >= 2:
                            days.append(future)
                            designs.append(bx)
                            caps.append(cx)
                            estimates.append(eligible)
                    if days:
                        fitted, residuals = cross_section_batch(
                            returns[days],
                            np.asarray(designs),
                            np.asarray(caps),
                            np.asarray(estimates),
                            np.arange(1, len(c.industry_codes) + 2),
                        )
                        for j, future in enumerate(days):
                            industry = designs[j][:, 1 : len(c.industry_codes) + 2]
                            singleton = industry[estimates[j]].sum(axis=0) == 1
                            residuals[j][
                                estimates[j] & industry[:, singleton].any(axis=1)
                            ] = np.nan
                            regressed[future] = fitted[j], residuals[j]
            b, names, est, cap = prepared.pop(day)
            k, n = b.shape[1], len(cap)
            f, residual = np.full(k, np.nan), np.full(n, np.nan)
            biases = np.array([np.nan, np.nan])
            if previous is not None and kind == "fundamental":
                prev_model, prev_state = previous
                if prev_model is None:
                    prev_model, _ = cache.load(
                        previous_key,
                        exposure_loader=lambda day=day: cached_exposure(day - 1),
                    )
                eligible = prev_state["estimation"].astype(bool) & np.isfinite(
                    returns[day]
                )
                if day in regressed:
                    f, residual = regressed.pop(day)
                    biases = np.array(
                        [
                            cross_sectional_bias(f, prev_state["base_f"]),
                            cross_sectional_bias(
                                residual,
                                prev_state["base_d"],
                                np.where(eligible, prev_state["cap"], 0.0),
                            ),
                        ]
                    )
            fs = np.asarray((history_f + [f])[-c.history :])
            if moments is not None:
                moments.push(day, residual)
            else:
                residual_window.push(day, residual)
            good_f = np.isfinite(fs).all(axis=1)
            if good_f.sum() >= c.min_descriptor_days:
                # 不压缩时间轴：没有回归的日期保留 NaN，NW 的 lag 仍是交易日。
                if moments is None:
                    rs = residual_window.window(max(0, day - c.history + 1), day)
                    covariance, d, base_f, base_d = estimate_covariances(
                        fs, rs, b, cap, c, est
                    )
                else:
                    covariance, d, base_f, base_d = estimate_covariances(
                        fs, moments.variances(day), b, cap, c, est
                    )
            else:
                covariance = np.eye(k) * c.cold_factor_variance
                d = np.full(n, c.cold_specific_variance)
                base_f, base_d = np.diag(covariance).copy(), d.copy()
            if c.regime:
                bs = np.asarray((history_bias + [biases])[-c.history :])
                for col in range(2):
                    valid = np.isfinite(bs[:, col])
                    if valid.any():
                        weights = exp_weights(len(bs), c.regime_half_life) * valid
                        multiplier = (
                            np.dot(weights, np.nan_to_num(bs[:, col])) / weights.sum()
                        )
                        if col == 0:
                            covariance *= max(multiplier, 1e-6)
                        else:
                            d *= max(multiplier, 1e-6)
            metadata = {
                "kind": kind,
                "version": source.version,
                "parameter_hash": params,
                "prefix_hash": prefix.hex(),
                "estimation_count": int(est.sum()),
                "asof": "close; forecasts next trading day",
                "regime": c.regime,
            }
            model = RiskModel(
                tuple(source.symbols),
                b,
                names,
                covariance,
                d,
                str(source.dates[day]),
                metadata,
            )
            if kind == "statistical" and day >= c.min_descriptor_days:
                model = statistical_for(day)
                model.metadata.update(metadata)
            state = {
                "f": f,
                "residual": residual,
                "base_f": base_f,
                "base_d": base_d,
                "bias": biases,
                "estimation": est,
                "cap": cap,
            }
            if cache:
                cache.save(
                    key,
                    model,
                    state,
                    day=day,
                    latest=day == end - 1,
                    checkpoint=rolling.checkpoint() if day == end - 1 else None,
                )
        if residual_window.latest < day:
            if moments is not None:
                moments.push(day, state["residual"])
            else:
                residual_window.push(day, state["residual"])
        history_f.append(state["f"])
        history_bias.append(state["bias"])
        if len(history_f) > c.history:
            history_f.pop(0)
            history_bias.pop(0)
        previous = model, state
        previous_key = key
        if day >= start:
            yield model
