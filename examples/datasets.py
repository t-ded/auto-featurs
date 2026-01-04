from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional
import urllib.request

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / '_data'


@dataclass(frozen=True)
class DatasetSpec:
    url: str
    filename: str


_DATASETS: dict[str, DatasetSpec] = {
    'bikes': DatasetSpec(
        url='TODO',
        filename='bikes.csv',
    ),
}


def get_dataset(name: str, url: Optional[str] = None) -> Path:
    """
    Returns a local path to the dataset, downloading it if necessary.
    """
    DATA_DIR.mkdir(exist_ok=True)

    spec = _DATASETS.get(name)
    if spec is not None and url is not None:
        logger.warning(
            "URL was provided for a built-in dataset '%s' and will be ignored.",
            name,
        )

    if spec is None:
        if url is None:
            raise ValueError(f"Unknown dataset '{name}' without source URL.")
        spec = DatasetSpec(url=url, filename=name)

    path = DATA_DIR / spec.filename
    _download_dataset(path, spec.url)
    return path


def _download_dataset(path: Path, url: str) -> None:
    if path.exists():
        return

    logger.info('Downloading dataset from %s', url)
    urllib.request.urlretrieve(url, path)

    head = path.read_text(encoding='utf-8', errors='ignore')[:200].lower()
    if '<html' in head:
        raise RuntimeError(
            f'Downloaded HTML instead of dataset from {url}. '
            'Make sure you are using a raw content URL '
            '(e.g. raw.githubusercontent.com).'
        )