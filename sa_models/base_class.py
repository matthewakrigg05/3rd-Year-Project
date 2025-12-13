from abc import ABC, abstractmethod
from typing import Dict

from sa_models.sentiment_result import SentimentResult


# Abstract base class for the analysis methods
class SentimentAnalyserBase(ABC):
    """
    A base class for all the sentiment analyser classes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyses the text and returns the sentiment score, alongside the original text.
        :param text:
        :return:
        """
        pass

    @staticmethod
    def score_to_label(
        score: float,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05
    ) -> str:
        """
        Convert a continuous sentiment score into a label.

        Default thresholds are aligned with VADER conventions,
        but can be overridden if required.
        """
        if score >= pos_threshold:
            return "positive"
        elif score <= neg_threshold:
            return "negative"
        return "neutral"


    @staticmethod
    def validate_result(result: SentimentResult) -> None:
        """
        Validate output format (useful for debugging and tests).
        """
        if not isinstance(result.text, str):
            raise TypeError("Result.text must be a string")

        if not isinstance(result.score, (float, int)):
            raise TypeError("Result.score must be numeric")

        if result.label not in {"positive", "neutral", "negative"}:
            raise ValueError("Invalid sentiment label")