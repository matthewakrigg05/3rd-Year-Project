from pathlib import Path
import pandas as pd
from data_cleaning.data_cleaning_functions import preprocess_tweet
from utils.data_loading import load_csv_to_dataframe


def clean_tweets(
    input_path: Path | str = "collected_tweets.csv",
    output_path: Path | str = "cleaned_tweets.csv",
    text_column: str = "text",
    keep_original: bool = True,
    dedupe_on: str = "cleaned_text",
):
    """Read tweets, preprocess text, dedupe, and write cleaned CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = load_csv_to_dataframe(input_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input {input_path}")

    # Apply preprocessing
    df["cleaned_text"] = df[text_column].astype(str).apply(preprocess_tweet)

    # Drop duplicates from the cleaned text; keep first occurrence, preserve all present columns
    if dedupe_on not in df.columns:
        raise ValueError(f"Deduplication column '{dedupe_on}' not found after cleaning")

    df = df.drop_duplicates(subset=dedupe_on, keep="first").reset_index(drop=True)

    # Keep original text optionally, otherwise drop it.
    if not keep_original and text_column in df.columns:
        df = df.drop(columns=[text_column])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    return df


def main() -> None:
    input_path = "collected_tweets.csv"
    output_path = "cleaned_tweets.csv"
    text_column = "text"

    print(f"Loading tweets from: {input_path}")
    print(f"Writing cleaned deduplicated tweets to: {output_path}")

    df_cleaned = clean_tweets(
        input_path=input_path,
        output_path=output_path,
        text_column=text_column,
        keep_original=True,
    )

    print(f"Cleaned tweets = {len(df_cleaned)} rows (deduped by cleaned_text)")


if __name__ == "__main__":
    main()
