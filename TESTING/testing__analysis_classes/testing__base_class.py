import unittest
from sa_models.base_class import SentimentAnalyserBase
from sa_models.sentiment_result import SentimentResult


class testing__base_class(unittest.TestCase):

    # # # # # # # # # #
    # score_to_label  #
    # # # # # # # # # #
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

    # # # # # # # # # #
    # validate_result #
    # # # # # # # # # #

    def test_validate_result(self):
        result = SentimentResult(
            text="hello",
            score=0.3,
            label="positive"
        )

        SentimentAnalyserBase.validate_result(result)

    def test_validate_result_invalid_label(self):
        result = SentimentResult(
            text="hello",
            score=0.3,
            label="britain"
        )

        with self.assertRaises(ValueError):
            SentimentAnalyserBase.validate_result(result)

    def test_validate_result_text(self):
        result = SentimentResult(
            text=42,
            score=0.3,
            label="positive"
        )

        with self.assertRaises(TypeError):
            SentimentAnalyserBase.validate_result(result)

    def test_validate_result_score(self):
        result = SentimentResult(
            text="hello",
            score="banana",
            label="positive"
        )

        with self.assertRaises(TypeError):
            SentimentAnalyserBase.validate_result(result)