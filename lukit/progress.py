import sys
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def _ascii_progress(
    iterable: Iterable[T],
    desc: str,
    total: Optional[int],
) -> Iterator[T]:
    if total is not None and total <= 0:
        return iter(())

    width = 28
    count = 0
    for count, item in enumerate(iterable, start=1):
        if total is None:
            if count == 1 or count % 10 == 0:
                print(f"\r{desc}: {count}", end="", file=sys.stderr, flush=True)
        else:
            ratio = min(1.0, count / total)
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\r{desc}: [{bar}] {count}/{total}",
                end="",
                file=sys.stderr,
                flush=True,
            )
        yield item
    if count > 0:
        print("", file=sys.stderr, flush=True)


def wrap_progress(
    iterable: Iterable[T],
    desc: str,
    total: Optional[int] = None,
) -> Iterator[T]:
    try:
        from tqdm.auto import tqdm

        return iter(tqdm(iterable, desc=desc, total=total, dynamic_ncols=True))
    except Exception:
        return _ascii_progress(iterable, desc=desc, total=total)
