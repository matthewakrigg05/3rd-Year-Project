from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    """
    Standardised sentiment analysis output.
    """
    text: str
    score: float
    label: str