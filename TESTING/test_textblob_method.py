"""
Unit tests for TextBlob sentiment analysis model.

@author: MA
"""
import unittest
import time
import pandas as pd
from src.models.textblob_method import TextBlobSentimentAnalyser
from src.models.sentiment_result import SentimentResult


class TestTextBlobInitialization(unittest.TestCase):
    """Test TextBlob model initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = TextBlobSentimentAnalyser()

    def test_initialization(self):
        """Test that TextBlob model initializes correctly."""
        self.assertEqual(self.analyser.name, "TextBlob")

    def test_analyzer_type(self):
        """Test that analyser is correctly instantiated."""
        self.assertIsNotNone(self.analyser)


class TestTextBlobSinglePrediction(unittest.TestCase):
    """Test TextBlob single text prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = TextBlobSentimentAnalyser()
        self.positive_texts = [
            "This is wonderful and beautiful!",
            "I love this so much!",
            "Excellent and fantastic!",
            "Best thing ever",
            "Very happy and joyful"
        ]
        self.negative_texts = [
            "This is horrible and ugly",
            "I dislike this very much",
            "Terrible and disgusting",
            "Worst thing ever",
            "Very sad and depressed"
        ]
        self.neutral_texts = [
            "The color is red",
            "The number is five",
            "The time is noon"
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

    def test_neutral_sentiment(self):
        """Test prediction on neutral text."""
        for text in self.neutral_texts:
            result = self.analyser.analyse(text)
            self.assertIsNotNone(result)
            self.assertIsInstance(result["score"], float)

    def test_score_range(self):
        """Test that score is in valid range [-1, 1]."""
        test_text = "This is a test"
        result = self.analyser.analyse(test_text)
        self.assertGreaterEqual(result["score"], -1.0)
        self.assertLessEqual(result["score"], 1.0)


class TestTextBlobBatchPrediction(unittest.TestCase):
    """Test TextBlob batch prediction capability."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = TextBlobSentimentAnalyser()

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

    def test_batch_with_varied_lengths(self):
        """Test batch prediction with varied text lengths."""
        texts = [
            "Short",
            "Medium length text here",
            "This is a much longer text that contains multiple sentences. It has more words and complexity. This should still be analyzed correctly.",
            "One",
            "Another medium text"
        ]
        results = self.analyser.analyse_batch(texts)
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result["score"], float)


class TestTextBlobErrorHandling(unittest.TestCase):
    """Test TextBlob error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = TextBlobSentimentAnalyser()

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


class TestTextBlobPerformance(unittest.TestCase):
    """Test TextBlob performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = TextBlobSentimentAnalyser()

    def test_single_prediction_timing(self):
        """Test that single prediction is reasonably fast."""
        text = "This is a test sentence for timing"
        
        start_time = time.time()
        self.analyser.analyse(text)
        elapsed = time.time() - start_time
        
        # Should be relatively fast
        self.assertLess(elapsed, 5.0)
        print(f"TextBlob single prediction time: {elapsed:.3f}s")

    def test_batch_prediction_timing(self):
        """Test batch prediction performance."""
        texts = ["Test text"] * 20
        
        start_time = time.time()
        self.analyser.analyse_batch(texts)
        elapsed = time.time() - start_time
        
        # Should process 20 texts reasonably quickly
        self.assertLess(elapsed, 10.0)
        print(f"TextBlob batch (20 texts) time: {elapsed:.3f}s")


class TestTextBlobConsistency(unittest.TestCase):
    """Test TextBlob consistency."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = TextBlobSentimentAnalyser()

    def test_consistent_results(self):
        """Test that same text produces consistent results."""
        text = "I like this"
        result1 = self.analyser.analyse(text)
        result2 = self.analyser.analyse(text)
        
        self.assertEqual(result1["score"], result2["score"])
        self.assertEqual(result1["label"], result2["label"])

    def test_polarity_bounds(self):
        """Test that polarity is bounded correctly."""
        texts = ["very good", "very bad", "neutral"]
        for text in texts:
            result = self.analyser.analyse(text)
            self.assertGreaterEqual(result["score"], -1.0)
            self.assertLessEqual(result["score"], 1.0)


class TestTextBlobWithRealData(unittest.TestCase):
    """Test TextBlob with real tweet data."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = TextBlobSentimentAnalyser()

    def test_real_tweets_analysis(self):
        """Test analysis on real tweets from dataset."""
        try:
            df = pd.read_csv("cleaned_tweets.csv")
            sample_texts = df["cleaned_text"].head(10).tolist()
            
            for text in sample_texts:
                result = self.analyser.analyse(text)
                self.assertIsNotNone(result)
                self.assertIn(result["label"], ["positive", "neutral", "negative"])
        except FileNotFoundError:
            self.skipTest("cleaned_tweets.csv not found")


if __name__ == "__main__":
    unittest.main()
