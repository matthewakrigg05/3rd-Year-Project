"""
Unit tests for BERT sentiment analysis model.
"""
import unittest
import time
import pandas as pd
from src.models.bert_method import BertSentimentAnalyser
from src.models.sentiment_result import SentimentResult


class TestBertInitialization(unittest.TestCase):
    """Test BERT model initialisation."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = BertSentimentAnalyser()

    def test_initialization(self):
        """Test that BERT model initialises correctly."""
        self.assertIsNotNone(self.analyser.tokenizer)
        self.assertIsNotNone(self.analyser.model)
        self.assertEqual(self.analyser.name, "BERT")

    def test_device_availability(self):
        """Test that device is correctly set."""
        self.assertIn(self.analyser.device.type, ["cpu", "cuda"])


class TestBertSinglePrediction(unittest.TestCase):
    """Test BERT single text prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = BertSentimentAnalyser()
        self.positive_texts = [
            "This is amazing and wonderful!",
            "I love this product, it's excellent!",
            "Best day ever, so happy!"
        ]
        self.negative_texts = [
            "This is terrible and awful",
            "I hate this, very disappointed",
            "Worst experience ever"
        ]
        self.neutral_texts = [
            "The weather is today",
            "I went to the store",
            "This is a sentence"
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
            self.assertIn("text", result)
            self.assertIn("score", result)
            self.assertIn("label", result)
            self.assertIsInstance(result["score"], float)

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


class TestBertBatchPrediction(unittest.TestCase):
    """Test BERT batch prediction capability."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = BertSentimentAnalyser()
        self.texts = [
            "This is great!",
            "This is terrible",
            "This is neutral",
            "Excellent work!"
        ]

    def test_batch_prediction_structure(self):
        """Test batch prediction returns correct structure."""
        # Read sample from cleaned_tweets.csv
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

    def test_batch_with_mock_data(self):
        """Test batch prediction with mock data."""
        texts = ["Good", "Bad", "Neutral"]
        results = self.analyser.analyse_batch(texts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result["score"], float)


class TestBertErrorHandling(unittest.TestCase):
    """Test BERT error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = BertSentimentAnalyser()

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


class TestBertPerformance(unittest.TestCase):
    """Test BERT performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = BertSentimentAnalyser()

    def test_single_prediction_timing(self):
        """Test that single prediction completes in reasonable time."""
        text = "This is a test sentence for performance measurement."
        
        start_time = time.time()
        self.analyser.analyse(text)
        elapsed = time.time() - start_time
        
        # Should complete within 30 seconds (generous timeout for cold start)
        self.assertLess(elapsed, 30.0)
        print(f"Single prediction time: {elapsed:.3f}s")


class TestBertValidation(unittest.TestCase):
    """Test BERT output validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = BertSentimentAnalyser()

    def test_result_validation(self):
        """Test that all results pass validation."""
        texts = ["Good", "Bad", "Neutral"]
        for text in texts:
            result = self.analyser.analyse(text)
            # validate_result should not raise exception
            from src.models.sentiment_result import SentimentResult
            result_obj = SentimentResult(
                text=result["text"],
                score=result["score"],
                label=result["label"]
            )
            self.analyser.validate_result(result_obj)


if __name__ == "__main__":
    unittest.main()
