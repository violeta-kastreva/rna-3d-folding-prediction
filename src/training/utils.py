from pathlib import Path
from typing import Iterable


def join(root: Path, filenames: Iterable[str]) -> list[str]:
    return [(root / filename).as_posix() for filename in filenames]