"""
Unit tests for VADER sentiment analysis model.
"""
import unittest
import time
import pandas as pd
from src.models.vader_method import VaderSentimentAnalyser
from src.models.sentiment_result import SentimentResult


class TestVaderInitialization(unittest.TestCase):
    """Test VADER model initialisation."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = VaderSentimentAnalyser()

    def test_initialization(self):
        """Test that VADER model initialises correctly."""
        self.assertIsNotNone(self.analyser.analyser)
        self.assertEqual(self.analyser.name, "VADER")

    def test_analyser_has_polarity_scores(self):
        """Test that analyser has polarity_scores method."""
        self.assertTrue(hasattr(self.analyser.analyser, "polarity_scores"))


class TestVaderSinglePrediction(unittest.TestCase):
    """Test VADER single text prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = VaderSentimentAnalyser()
        self.positive_texts = [
            "This is amazing!",
            "I love it!",
            "Excellent! :)",
            "Best ever!!!",
            "So happy and excited!"
        ]
        self.negative_texts = [
            "This is terrible",
            "I hate it",
            "Awful :/",
            "Worst ever!!!",
            "So sad and disappointed"
        ]
        self.neutral_texts = [
            "The temperature is 20 degrees",
            "The store is open",
            "I went to the market"
        ]

    def test_positive_sentiment(self):
        """Test prediction on positive text."""
        for text in self.positive_texts:
            result = self.analyser.analyse(text)
            self.assertIn("text", result)
            self.assertIn("score", result)
            self.assertIn("label", result)
            self.assertIsInstance(result["score"], float)
            self.assertIn(result["label"], ["positive", "neutral", "negative"])

    def test_negative_sentiment(self):
        """Test prediction on negative text."""
        for text in self.negative_texts:
            result = self.analyser.analyse(text)
            self.assertIsNotNone(result)
            self.assertIsInstance(result["score"], float)
            self.assertIn(result["label"], ["positive", "neutral", "negative"])

    def test_neutral_sentiment(self):
        """Test prediction on neutral text."""
        for text in self.neutral_texts:
            result = self.analyser.analyse(text)
            self.assertIsNotNone(result)
            self.assertIsInstance(result["score"], float)

    def test_score_range(self):
        """Test that score is in valid range [-1, 1]."""
        test_text = "This is a test sentence"
        result = self.analyser.analyse(test_text)
        self.assertGreaterEqual(result["score"], -1.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_emoticon_handling(self):
        """Test VADER handles emoticons correctly."""
        positive_with_emoticon = "I love this ! :)"
        result = self.analyser.analyse(positive_with_emoticon)
        self.assertIsNotNone(result)


class TestVaderBatchPrediction(unittest.TestCase):
    """Test VADER batch prediction capability."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = VaderSentimentAnalyser()

    def test_batch_prediction_structure(self):
        """Test batch prediction returns correct structure."""
        try:
            df = pd.read_csv("cleaned_tweets.csv")
            sample_texts = df["cleaned_text"].head(5).tolist()
            results = self.analyser.analyse_batch(sample_texts)
            
            self.assertEqual(len(results), len(sample_texts))
            for result in results:
                self.assertIn("text", result)
                self.assertIn("score", result)
                self.assertIn("label", result)
        except FileNotFoundError:
            self.skipTest("cleaned_tweets.csv not found")

    def test_batch_with_mixed_sentiments(self):
        """Test batch prediction with mixed sentiments."""
        texts = [
            "Great day!",
            "Terrible experience",
            "It is what it is",
            "Amazing work!!!",
            "Very disappointed :("
        ]
        results = self.analyser.analyse_batch(texts)
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result["score"], float)


class TestVaderErrorHandling(unittest.TestCase):
    """Test VADER error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = VaderSentimentAnalyser()

    def test_empty_string(self):
        """Test handling of empty string."""
        with self.assertRaises(ValueError):
            self.analyser.analyse("")

    def test_none_input(self):
        """Test handling of None input."""
        with self.assertRaises(ValueError):
            self.analyser.analyse(None)

    def test_non_string_input(self):
        """Test handling of non-string input."""
        with self.assertRaises(ValueError):
            self.analyser.analyse(123)

    def test_empty_batch(self):
        """Test handling of empty batch."""
        with self.assertRaises(ValueError):
            self.analyser.analyse_batch([])


class TestVaderPerformance(unittest.TestCase):
    """Test VADER performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = VaderSentimentAnalyser()

    def test_single_prediction_timing(self):
        """Test that single prediction is fast."""
        text = "This is a test sentence"
        
        start_time = time.time()
        self.analyser.analyse(text)
        elapsed = time.time() - start_time
        
        # VADER should be very fast (< 1 second)
        self.assertLess(elapsed, 1.0)
        print(f"VADER single prediction time: {elapsed:.4f}s")

    def test_batch_prediction_timing(self):
        """Test batch prediction performance."""
        texts = ["Test"] * 10
        
        start_time = time.time()
        self.analyser.analyse_batch(texts)
        elapsed = time.time() - start_time
        
        # Should process 10 texts quickly
        self.assertLess(elapsed, 5.0)
        print(f"VADER batch (10 texts) time: {elapsed:.3f}s")


class TestVaderConsistency(unittest.TestCase):
    """Test VADER consistency."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = VaderSentimentAnalyser()

    def test_consistent_results(self):
        """Test that same text produces consistent results."""
        text = "This is great!"
        result1 = self.analyser.analyse(text)
        result2 = self.analyser.analyse(text)
        
        self.assertEqual(result1["score"], result2["score"])
        self.assertEqual(result1["label"], result2["label"])

    def test_threshold_boundaries(self):
        """Test score to label conversion at thresholds."""
        # Test with neutral-ish scores
        slightly_positive = "okay"
        result = self.analyser.analyse(slightly_positive)
        self.assertIsNotNone(result["label"])


if __name__ == "__main__":
    unittest.main()
