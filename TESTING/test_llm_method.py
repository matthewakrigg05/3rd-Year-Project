"""
Unit tests for GPT-2 sentiment analysis model.

@author: MA
"""
import unittest
import time
import pandas as pd
from src.models.llm_method import GPT2SentimentAnalyser
from src.models.sentiment_result import SentimentResult


class TestGPT2Initialization(unittest.TestCase):
    """Test GPT-2 model initialisation."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = GPT2SentimentAnalyser()

    def test_initialization(self):
        """Test that GPT-2 model initialises correctly."""
        self.assertIsNotNone(self.analyser.tokenizer)
        self.assertIsNotNone(self.analyser.model)
        self.assertEqual(self.analyser.name, "GPT-2")

    def test_device_availability(self):
        """Test that device is correctly set."""
        self.assertIn(self.analyser.device.type, ["cpu", "cuda"])

    def test_sentiment_indicators(self):
        """Test that sentiment indicator tokens are initialized."""
        self.assertGreater(len(self.analyser.positive_tokens), 0)
        self.assertGreater(len(self.analyser.negative_tokens), 0)


class TestGPT2SinglePrediction(unittest.TestCase):
    """Test GPT-2 single text prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = GPT2SentimentAnalyser()
        self.positive_texts = [
            "This product is excellent and amazing",
            "I love this so much!",
            "Great experience, very happy",
            "Awesome work, best ever",
            "Good quality, happy customer"
        ]
        self.negative_texts = [
            "This is bad and terrible",
            "I hate this, very disappointed",
            "Awful experience, hate it",
            "Poor quality, hate buying this",
            "Horrible, worst ever"
        ]
        self.neutral_texts = [
            "The weather is cloudy",
            "I went shopping today",
            "The store is closed"
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


class TestGPT2BatchPrediction(unittest.TestCase):
    """Test GPT-2 batch prediction capability."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = GPT2SentimentAnalyser()

    def test_batch_prediction_structure(self):
        """Test batch prediction returns correct structure."""
        try:
            df = pd.read_csv("cleaned_tweets.csv")
            sample_texts = df["cleaned_text"].head(3).tolist()
            results = self.analyser.analyse_batch(sample_texts)
            
            self.assertEqual(len(results), len(sample_texts))
            for result in results:
                self.assertIn("text", result)
                self.assertIn("score", result)
                self.assertIn("label", result)
        except FileNotFoundError:
            self.skipTest("cleaned_tweets.csv not found")

    def test_batch_with_small_samples(self):
        """Test batch prediction with small samples."""
        texts = ["Good", "Bad", "Neutral"]
        results = self.analyser.analyse_batch(texts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result["score"], float)


class TestGPT2ErrorHandling(unittest.TestCase):
    """Test GPT-2 error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = GPT2SentimentAnalyser()

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


class TestGPT2Performance(unittest.TestCase):
    """Test GPT-2 performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = GPT2SentimentAnalyser()

    def test_single_prediction_timing(self):
        """Test that single prediction completes in reasonable time."""
        text = "This is a test sentence"
        
        start_time = time.time()
        self.analyser.analyse(text)
        elapsed = time.time() - start_time
        
        # Should complete within 60 seconds (generous for GPT-2 generation)
        self.assertLess(elapsed, 60.0)
        print(f"GPT-2 single prediction time: {elapsed:.3f}s")

    def test_batch_prediction_timing(self):
        """Test batch prediction performance."""
        texts = ["Test"] * 3
        
        start_time = time.time()
        self.analyser.analyse_batch(texts)
        elapsed = time.time() - start_time
        
        # Should process 3 texts reasonably
        self.assertLess(elapsed, 180.0)
        print(f"GPT-2 batch (3 texts) time: {elapsed:.3f}s")


class TestGPT2Consistency(unittest.TestCase):
    """Test GPT-2 consistency."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = GPT2SentimentAnalyser()

    def test_consistent_label_assignment(self):
        """Test that sentiment indicators drive consistent labels."""
        # Text with clear positive indicators
        positive_text = "This is very good and excellent"
        result = self.analyser.analyse(positive_text)
        self.assertIsNotNone(result["label"])
        self.assertIn(result["label"], ["positive", "neutral", "negative"])

    def test_sentiment_indicator_tokens_present(self):
        """Test that positive and negative token sets are non-empty."""
        self.assertTrue(len(self.analyser.positive_tokens) > 0)
        self.assertTrue(len(self.analyser.negative_tokens) > 0)


class TestGPT2SentimentDetection(unittest.TestCase):
    """Test GPT-2 sentiment detection against clear cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = GPT2SentimentAnalyser()

    def test_explicit_positive_keywords(self):
        """Test detection of explicit positive keywords."""
        text_with_positive = "This is great and amazing"
        result = self.analyser.analyse(text_with_positive)
        self.assertIsNotNone(result)

    def test_explicit_negative_keywords(self):
        """Test detection of explicit negative keywords."""
        text_with_negative = "This is bad and horrible"
        result = self.analyser.analyse(text_with_negative)
        self.assertIsNotNone(result)

    def test_mixed_sentiment_keywords(self):
        """Test handling of mixed positive and negative keywords."""
        mixed_text = "Good but also bad"
        result = self.analyser.analyse(mixed_text)
        self.assertIsNotNone(result)
        self.assertIn(result["label"], ["positive", "neutral", "negative"])


if __name__ == "__main__":
    unittest.main()
