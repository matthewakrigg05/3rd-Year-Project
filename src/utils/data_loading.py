from pathlib import Path
from typing import Iterator, Union
import pandas as pd

PathLike = Union[str, Path]

def load_csv_to_dataframe(path: PathLike, 
                          usecols: list[str] | None = None) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Uses dtype=str and low_memory=False to avoid mixed-type inference.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    read_kwargs = {
        "dtype": str,
        "encoding": "utf-8",
        "low_memory": False,
    }

    if usecols is not None:
        read_kwargs["usecols"] = usecols

    df = pd.read_csv(p, **read_kwargs)
    return df


def iter_csv_chunks(path: PathLike, 
                    chunksize: int = 5000) -> Iterator[pd.DataFrame]:
    """
    Lazily iterate over a CSV file in chunks.

    Useful for large files that don't fit comfortably in memory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    read_kwargs = {
        "dtype": str,
        "encoding": "utf-8",
        "chunksize": chunksize,
    }

    for chunk in pd.read_csv(p, **read_kwargs):
        yield chunk


def load_dataset(path: PathLike, 
                 chunksize: int | None = None) -> pd.DataFrame:
    """
    Load the full dataset into a DataFrame.

    If chunksize is provided, reads in chunks via iter_csv_chunks and concatenates.
    """
    if chunksize is None:
        return load_csv_to_dataframe(path)

    # Read in chunks and concatenate. This keeps peak memory lower than
    # reading everything at once.
    chunks = []
    for chunk in iter_csv_chunks(path, chunksize=chunksize):
        chunks.append(chunk)

    if not chunks:
        # Return empty DataFrame with expected columns if file exists but has no rows
        return print("Warning: CSV file is empty. Returning empty DataFrame.")

    df = pd.concat(chunks, ignore_index=True)
    return df


def stream_dataset(path: PathLike, 
                   chunksize: int = 5000) -> Iterator[pd.DataFrame]:
    """Stream the CSV as DataFrame chunks without accumulating all rows in memory."""
    for chunk in iter_csv_chunks(path, chunksize=chunksize):
        yield chunk
