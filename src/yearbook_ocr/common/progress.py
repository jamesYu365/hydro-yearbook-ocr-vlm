from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress(
    items: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
    unit: str = "item",
    disable: bool = False,
) -> Iterator[T]:
    return tqdm(items, desc=desc, total=total, unit=unit, disable=disable)
