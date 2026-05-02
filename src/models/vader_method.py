"""
VADER sentiment analysis for social media and informal text.

VADER is a lexicon and rule-based tool optimised for short, informal text.
It produces a compound score in [-1, 1] without any model training.
"""
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class VaderSentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    A lexicon and rule-based tool that works well for social media text, handling
    emoticons, slang, and negations without any model training.
    """

    def __init__(self):
        """Initialises the VADER SentimentIntensityAnalyzer."""
        self.analyser = SentimentIntensityAnalyzer()

    @property
    def name(self) -> str:
        """Return the model identifier."""
        return "VADER"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse sentiment of the given text using VADER.
        
        Returns a dict with 'text', 'score' (compound, in [-1, 1]), and 'label'.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            scores = self.analyser.polarity_scores(text)
            compound_score = scores["compound"]
            label = self.score_to_label(
                compound_score,
                pos_threshold=0.05,
                neg_threshold=-0.05
            )

            result_obj = SentimentResult(
                text=text,
                score=compound_score,
                label=label
            )
            self.validate_result(result_obj)
            return {
                "text": result_obj.text,
                "score": result_obj.score,
                "label": result_obj.label
            }

        except Exception as e:
            raise RuntimeError(f"VADER analysis failed: {str(e)}")

    def analyse_batch(self, texts: list) -> list:
        """Analyse a list of texts and return a list of result dicts."""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        results = []
        for text in texts:
            results.append(self.analyse(text))
        return results
