import unittest
from sa_models.base_class import SentimentAnalyserBase

class testing__base_class(unittest.TestCase):

    """
    Testing the static score_to_label function, expecting to classify the scores
    passed to it. Function has customisable thresholds for classification, and should
    react accordingly.
    """
    def test_score_to_label_positive(self):
        result = SentimentAnalyserBase.score_to_label(score=0.6)
        self.assertEqual(result, "positive")

    def test_score_to_label_negative(self):
        result = SentimentAnalyserBase.score_to_label(score=-0.6)
        self.assertEqual(result, "negative")

    def test_score_to_label_neutral(self):
        result = SentimentAnalyserBase.score_to_label(score=0.0)
        self.assertEqual(result, "neutral")

    def test_score_to_label_custom_thresholds(self):
        result = SentimentAnalyserBase.score_to_label(
            0.1, pos_threshold=0.2
        )
        self.assertEqual(result, "neutral")