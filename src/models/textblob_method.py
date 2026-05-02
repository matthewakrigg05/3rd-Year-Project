"""
TextBlob sentiment analysis.

Uses the TextBlob library to produce a polarity score in [-1, 1] and
a sentiment label (positive, neutral, or negative).
"""
from typing import Dict
from textblob import TextBlob

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class TextBlobSentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using the TextBlob library.
    
    TextBlob uses a pattern-based classifier to score text polarity
    from -1 (negative) to 1 (positive).
    """

    def __init__(self):
        """No initialisation needed for TextBlob."""
        pass

    @property
    def name(self) -> str:
        """Return the model identifier."""
        return "TextBlob"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse sentiment of the given text using TextBlob.
        
        Returns a dict with 'text', 'score' (in [-1, 1]), and 'label'.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            blob = TextBlob(text)
            polarity_score = blob.sentiment.polarity
            label = self.score_to_label(
                polarity_score,
                pos_threshold=0.05,
                neg_threshold=-0.05
            )

            result_obj = SentimentResult(
                text=text,
                score=polarity_score,
                label=label
            )
            self.validate_result(result_obj)
            return {
                "text": result_obj.text,
                "score": result_obj.score,
                "label": result_obj.label
            }

        except Exception as e:
            raise RuntimeError(f"TextBlob analysis failed: {str(e)}")

    def analyse_batch(self, texts: list) -> list:
        """Analyse a list of texts and return a list of result dicts."""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        results = []
        for text in texts:
            results.append(self.analyse(text))
        return results
