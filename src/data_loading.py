from pathlib import Path
from typing import Iterator, Union
import pandas as pd

PathLike = Union[str, Path]

def load_csv_to_dataframe(path: PathLike, 
                          usecols: list[str] | None = None) -> pd.DataFrame:
    """
    Load the CSV into a pandas DataFrame.

    Parameters:
    - path: path to the CSV file (string or Path)
    - usecols: optional list of column names to read (helps memory).

    Returns:
    - pandas.DataFrame

    Notes:
    - Uses `low_memory=False` and `dtype=str` to avoid mixed-type inference and
      to preserve text content.
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
                    chunksize: int = 10000, 
                    usecols: list[str] | None = None) -> Iterator[pd.DataFrame]:
    """
    Lazily iterate over the CSV file yielding DataFrame chunks.

    Parameters:
    - path: path to the CSV file
    - chunksize: number of rows per yielded DataFrame
    - usecols: optional list of columns to read

    Yields:
    - pandas.DataFrame of at most `chunksize` rows

    This is the recommended approach for processing large CSV files that
    don't fit comfortably into memory. Each chunk will have the same
    columns as read by pandas.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    read_kwargs = {
        "dtype": str,
        "encoding": "utf-8",
        "chunksize": chunksize,
    }
    if usecols is not None:
        read_kwargs["usecols"] = usecols

    for chunk in pd.read_csv(p, **read_kwargs):
        yield chunk


def load_dataset(path: PathLike, 
                 chunksize: int | None = None, 
                 usecols: list[str] | None = None) -> pd.DataFrame:
    """
    Load the whole dataset into a pandas DataFrame.

    If `chunksize` is None the function delegates to `load_csv_to_dataframe`.
    If `chunksize` is provided the CSV is read in chunks via
    `iter_csv_chunks` and concatenated. Use `chunksize` when the file is
    large but you still need a single DataFrame object; note this will
    temporarily hold each chunk in memory while concatenating.

    Parameters:
    - path: path to the CSV file
    - chunksize: optional chunk size for reading with `iter_csv_chunks`
    - usecols: optional column subset to load

    Returns:
    - pandas.DataFrame containing the entire dataset (excluding header)
    """
    if chunksize is None:
        return load_csv_to_dataframe(path, usecols=usecols)

    # Read in chunks and concatenate. This keeps peak memory lower than
    # reading everything at once.
    chunks = []
    for chunk in iter_csv_chunks(path, chunksize=chunksize, usecols=usecols):
        chunks.append(chunk)

    if not chunks:
        # Return empty DataFrame with expected columns if file exists but has no rows
        return pd.DataFrame(columns=usecols if usecols is not None else [])

    df = pd.concat(chunks, ignore_index=True)
    return df


def stream_dataset(path: PathLike, chunksize: int = 10000, usecols: list[str] | None = None) -> Iterator[pd.DataFrame]:
    """
    Stream the CSV dataset as DataFrame chunks. Yields each chunk from
    `iter_csv_chunks` without accumulating them in memory.

    Use this when processing can be done per-chunk (e.g., cleaning,
    feature extraction) and you want to avoid building a single large
    DataFrame.
    """
    for chunk in iter_csv_chunks(path, chunksize=chunksize, usecols=usecols):
        yield chunk
