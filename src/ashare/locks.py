"""跨模块共用的文件锁；数据层和研究台账都从这里取，不经数据层内部模块。"""

from contextlib import contextmanager


@contextmanager
def lock(path, *, shared=False, blocking=True):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(stream, mode | (0 if blocking else fcntl.LOCK_NB))
        yield
