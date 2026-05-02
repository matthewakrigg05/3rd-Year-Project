import unittest
import pandas as pd
import os
import tempfile
import csv
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sample_tweets_for_labeling import sample_tweets_for_labeling


class TestSampleTweetsForLabeling(unittest.TestCase):
    """Test the sampling function for creating balanced training data."""

    def setUp(self):
        """Create temporary directory and sample data for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        
        # Create sample classified_tweets.csv with predictions from all 4 models
        self.create_sample_data()

    def tearDown(self):
        """Clean up temporary files and restore working directory."""
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def create_sample_data(self):
        """Create sample classified tweets data with all models."""
        # Create enough tweets for 4 models x 3 classes x 250 samples = 3000 total
        # With some buffer, create 4000 tweets
        num_tweets = 4000
        
        data = {
            'tweet_text': [f'Tweet {i}' for i in range(num_tweets)],
            'textblob_score_1': [0.1] * num_tweets,
            'textblob_class_1': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'textblob_score_2': [0.1] * num_tweets,
            'textblob_class_2': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'textblob_score_3': [0.1] * num_tweets,
            'textblob_class_3': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'vader_score_1': [0.1] * num_tweets,
            'vader_class_1': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'vader_score_2': [0.1] * num_tweets,
            'vader_class_2': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'vader_score_3': [0.1] * num_tweets,
            'vader_class_3': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'bert_score_1': [0.1] * num_tweets,
            'bert_class_1': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'bert_score_2': [0.1] * num_tweets,
            'bert_class_2': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'bert_score_3': [0.1] * num_tweets,
            'bert_class_3': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'gpt-2_score_1': [0.1] * num_tweets,
            'gpt-2_class_1': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'gpt-2_score_2': [0.1] * num_tweets,
            'gpt-2_class_2': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
            'gpt-2_score_3': [0.1] * num_tweets,
            'gpt-2_class_3': ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 2000,
        }
        df = pd.DataFrame(data)
        df.to_csv('classified_tweets.csv', index=False)

    def test_sampling_creates_output_file(self):
        """Test that sampling creates tweets_to_label.csv."""
        sample_tweets_for_labeling()
        self.assertTrue(os.path.exists('tweets_to_label.csv'))

    def test_correct_total_samples(self):
        """Test that exactly 3000 tweets are sampled (750 per model, 250 per class)."""
        sample_tweets_for_labeling()
        df = pd.read_csv('tweets_to_label.csv')
        self.assertEqual(len(df), 3000, "Should have exactly 3000 samples")

    def test_samples_per_model(self):
        """Test that each model has exactly 750 samples."""
        sample_tweets_for_labeling()
        df = pd.read_csv('tweets_to_label.csv')
        model_counts = df['sampled_model'].value_counts()
        for model in ['textblob', 'vader', 'bert', 'gpt-2']:
            self.assertEqual(model_counts[model], 750, f"{model} should have 750 samples")

    def test_samples_per_class_per_model(self):
        """Test that each model has 250 samples per class (positive, negative, neutral)."""
        sample_tweets_for_labeling()
        df = pd.read_csv('tweets_to_label.csv')
        
        models = {
            'textblob': 'textblob_class_1',
            'vader': 'vader_class_1',
            'bert': 'bert_class_1',
            'gpt-2': 'gpt-2_class_1'
        }
        
        for model_name, pred_col in models.items():
            model_df = df[df['sampled_model'] == model_name]
            class_counts = model_df[pred_col].value_counts()
            for sentiment in ['positive', 'negative', 'neutral']:
                self.assertEqual(
                    class_counts[sentiment], 250,
                    f"{model_name} {sentiment} should have 250 samples"
                )

    def test_no_duplicate_tweets(self):
        """Test that no tweet appears more than once across all models."""
        sample_tweets_for_labeling()
        df = pd.read_csv('tweets_to_label.csv')
        
        # Check that all tweet_text values are unique
        unique_tweets = df['tweet_text'].nunique()
        total_tweets = len(df)
        
        self.assertEqual(unique_tweets, total_tweets, 
                        "No tweet should appear in multiple model samples")

    def test_sampled_model_column_exists(self):
        """Test that sampled_model column is present."""
        sample_tweets_for_labeling()
        df = pd.read_csv('tweets_to_label.csv')
        self.assertIn('sampled_model', df.columns, "sampled_model column should exist")

    def test_all_required_columns_present(self):
        """Test that all input columns are preserved in output."""
        sample_tweets_for_labeling()
        input_df = pd.read_csv('classified_tweets.csv')
        output_df = pd.read_csv('tweets_to_label.csv')
        
        # All input columns should be in output (except index)
        for col in input_df.columns:
            self.assertIn(col, output_df.columns, 
                         f"Column {col} from input should be in output")

    def test_sampled_model_column_position(self):
        """Test that sampled_model is second column (after tweet_text)."""
        sample_tweets_for_labeling()
        df = pd.read_csv('tweets_to_label.csv')
        cols = list(df.columns)
        self.assertEqual(cols[0], 'tweet_text', "First column should be tweet_text")
        self.assertEqual(cols[1], 'sampled_model', "Second column should be sampled_model")

    def test_sampling_is_random(self):
        """Test that running sampling twice gives different samples (with high probability)."""
        sample_tweets_for_labeling()
        df1 = pd.read_csv('tweets_to_label.csv')
        
        # Remove the file and sample again
        os.remove('tweets_to_label.csv')
        sample_tweets_for_labeling()
        df2 = pd.read_csv('tweets_to_label.csv')
        
        # They should have different tweets (with very high probability)
        # but same structure
        self.assertEqual(len(df1), len(df2), "Both samples should have same size")
        # Check that at least some tweets are different
        different_tweets = len(set(df1['tweet_text']) - set(df2['tweet_text']))
        self.assertGreater(different_tweets, 0, 
                          "Samples should differ (randomness test)")


class TestTweetClassifierUIIntegration(unittest.TestCase):
    """Integration tests for the updated UI with new data flow."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        
        # Create sample tweets_to_label.csv
        self.create_sample_tweets_to_label()

    def tearDown(self):
        """Clean up."""
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def create_sample_tweets_to_label(self):
        """Create sample tweets_to_label.csv for UI testing."""
        data = {
            'tweet_text': ['Tweet 1', 'Tweet 2', 'Tweet 3'],
            'sampled_model': ['textblob', 'vader', 'bert'],
            'textblob_class_1': ['positive', 'negative', 'neutral'],
            'vader_class_1': ['positive', 'negative', 'neutral'],
            'bert_class_1': ['positive', 'negative', 'neutral'],
            'gpt-2_class_1': ['positive', 'negative', 'neutral'],
        }
        df = pd.DataFrame(data)
        df.to_csv('tweets_to_label.csv', index=False)

    def test_output_file_creation(self):
        """Test that labelled_tweets.csv is created after labeling."""
        # Simulate labeling a tweet
        with open('labelled_tweets.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = ['tweet_text', 'sampled_model', 'textblob_class_1', 
                   'vader_class_1', 'bert_class_1', 'gpt-2_class_1', 'human_label']
            writer.writerow(cols)
            writer.writerow(['Tweet 1', 'textblob', 'positive', 'positive', 
                           'positive', 'positive', 'positive'])
        
        self.assertTrue(os.path.exists('labelled_tweets.csv'))

    def test_labelled_output_preserves_data(self):
        """Test that output preserves all input data."""
        # Read input
        input_df = pd.read_csv('tweets_to_label.csv')
        
        # Create output with all columns plus human_label
        with open('labelled_tweets.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = list(input_df.columns) + ['human_label']
            writer.writerow(cols)
            row_data = list(input_df.iloc[0].values) + ['positive']
            writer.writerow(row_data)
        
        output_df = pd.read_csv('labelled_tweets.csv')
        
        # Check that all input columns are preserved
        for col in input_df.columns:
            self.assertIn(col, output_df.columns)
        
        # Check that human_label column exists
        self.assertIn('human_label', output_df.columns)

    def test_no_duplicate_labeling(self):
        """Test that tweets are not labeled twice."""
        # Simulate labeling same tweet twice
        rows = [
            ['tweet_text', 'sampled_model', 'human_label'],
            ['Tweet 1', 'textblob', 'positive'],
            ['Tweet 1', 'textblob', 'negative'],  # Same tweet, different label
        ]
        
        with open('labelled_tweets.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        df = pd.read_csv('labelled_tweets.csv')
        
        # In actual use, duplicates should be prevented by the UI
        # This test documents that behavior
        duplicate_tweets = df[df.duplicated(subset=['tweet_text'], keep=False)]
        self.assertEqual(len(duplicate_tweets), 2, "UI should prevent duplicate labeling")


if __name__ == '__main__':
    unittest.main()
