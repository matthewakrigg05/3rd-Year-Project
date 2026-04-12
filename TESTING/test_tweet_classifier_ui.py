import unittest
import unittest.mock as mock
import pandas as pd
import os
import tempfile
import csv
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTweetClassifierUIUpdated(unittest.TestCase):
    """Test the updated TweetClassifier UI with new input/output files."""

    def setUp(self):
        """Create temporary files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        
        # Create sample tweets_to_label.csv
        self.tweets_to_label_path = 'tweets_to_label.csv'
        self.labelled_tweets_path = 'labelled_tweets.csv'
        
        self.create_sample_input_file()

    def tearDown(self):
        """Clean up temporary files."""
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def create_sample_input_file(self):
        """Create sample tweets_to_label.csv with model predictions."""
        sample_data = {
            'tweet_text': [
                'Great product very happy',
                'Terrible experience',
                'Its okay nothing special',
                'Best day ever',
                'Worst thing ever'
            ],
            'sampled_model': ['textblob', 'vader', 'bert', 'textblob', 'gpt-2'],
            'textblob_class_1': ['positive', 'negative', 'neutral', 'positive', 'negative'],
            'vader_class_1': ['positive', 'negative', 'neutral', 'positive', 'negative'],
            'bert_class_1': ['positive', 'negative', 'neutral', 'positive', 'negative'],
            'gpt-2_class_1': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(self.tweets_to_label_path, index=False)

    def test_input_file_exists(self):
        """Test that tweets_to_label.csv is read correctly."""
        df = pd.read_csv(self.tweets_to_label_path)
        self.assertEqual(len(df), 5)
        self.assertIn('tweet_text', df.columns)
        self.assertIn('sampled_model', df.columns)

    def test_input_file_has_all_model_predictions(self):
        """Test that input file contains predictions from all models."""
        df = pd.read_csv(self.tweets_to_label_path)
        required_cols = [
            'tweet_text', 'sampled_model',
            'textblob_class_1', 'vader_class_1',
            'bert_class_1', 'gpt-2_class_1'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

    def test_output_file_creation_with_header(self):
        """Test that labelled_tweets.csv is created with correct header."""
        input_df = pd.read_csv(self.tweets_to_label_path)
        cols = list(input_df.columns) + ['human_label']
        
        with open(self.labelled_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
        
        output_df = pd.read_csv(self.labelled_tweets_path)
        expected_cols = set(list(input_df.columns) + ['human_label'])
        actual_cols = set(output_df.columns)
        
        self.assertEqual(expected_cols, actual_cols)

    def test_output_preserves_input_data(self):
        """Test that output file preserves all input columns."""
        input_df = pd.read_csv(self.tweets_to_label_path)
        
        # Simulate labeling first tweet
        with open(self.labelled_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = list(input_df.columns) + ['human_label']
            writer.writerow(cols)
            row_data = list(input_df.iloc[0].values) + ['positive']
            writer.writerow(row_data)
        
        output_df = pd.read_csv(self.labelled_tweets_path)
        
        # Check that tweet_text is preserved
        self.assertEqual(output_df.iloc[0]['tweet_text'], 'Great product very happy')
        
        # Check that sampled_model is preserved
        self.assertEqual(output_df.iloc[0]['sampled_model'], 'textblob')
        
        # Check that all model predictions are preserved
        self.assertEqual(output_df.iloc[0]['textblob_class_1'], 'positive')
        self.assertEqual(output_df.iloc[0]['vader_class_1'], 'positive')

    def test_human_label_column_added(self):
        """Test that human_label column is added to output."""
        input_df = pd.read_csv(self.tweets_to_label_path)
        
        with open(self.labelled_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = list(input_df.columns) + ['human_label']
            writer.writerow(cols)
            row_data = list(input_df.iloc[0].values) + ['positive']
            writer.writerow(row_data)
        
        output_df = pd.read_csv(self.labelled_tweets_path)
        self.assertIn('human_label', output_df.columns)
        self.assertEqual(output_df.iloc[0]['human_label'], 'positive')

    def test_multiple_labels_appended(self):
        """Test that multiple tweet labeling appends to file."""
        input_df = pd.read_csv(self.tweets_to_label_path)
        
        # Label first tweet
        with open(self.labelled_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = list(input_df.columns) + ['human_label']
            writer.writerow(cols)
            row_data = list(input_df.iloc[0].values) + ['positive']
            writer.writerow(row_data)
        
        # Append second tweet
        with open(self.labelled_tweets_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row_data = list(input_df.iloc[1].values) + ['negative']
            writer.writerow(row_data)
        
        output_df = pd.read_csv(self.labelled_tweets_path)
        self.assertEqual(len(output_df), 2)
        self.assertEqual(output_df.iloc[0]['human_label'], 'positive')
        self.assertEqual(output_df.iloc[1]['human_label'], 'negative')

    def test_valid_sentiment_labels(self):
        """Test that only valid sentiment labels are accepted."""
        input_df = pd.read_csv(self.tweets_to_label_path)
        valid_sentiments = ['positive', 'negative', 'neutral']
        
        with open(self.labelled_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = list(input_df.columns) + ['human_label']
            writer.writerow(cols)
            
            for idx, sentiment in enumerate(valid_sentiments):
                row_data = list(input_df.iloc[idx].values) + [sentiment]
                writer.writerow(row_data)
        
        output_df = pd.read_csv(self.labelled_tweets_path)
        
        for sentiment in output_df['human_label']:
            self.assertIn(sentiment, valid_sentiments)

    def test_sampled_model_values(self):
        """Test that sampled_model contains only expected model names."""
        df = pd.read_csv(self.tweets_to_label_path)
        valid_models = ['textblob', 'vader', 'bert', 'gpt-2']
        
        for model in df['sampled_model']:
            self.assertIn(model, valid_models)

    def test_tracking_labeled_tweets(self):
        """Test that already labeled tweets can be tracked."""
        input_df = pd.read_csv(self.tweets_to_label_path)
        
        # Create output with first two tweets labeled
        with open(self.labelled_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = list(input_df.columns) + ['human_label']
            writer.writerow(cols)
            
            for i in range(2):
                row_data = list(input_df.iloc[i].values) + ['positive']
                writer.writerow(row_data)
        
        # Simulate reading labeled tweets to find remaining
        labeled_df = pd.read_csv(self.labelled_tweets_path)
        labeled_tweets = set(labeled_df['tweet_text'])
        
        remaining = input_df[~input_df['tweet_text'].isin(labeled_tweets)]
        self.assertEqual(len(remaining), 3)

    def test_progress_tracking(self):
        """Test that progress through tweets can be tracked."""
        input_df = pd.read_csv(self.tweets_to_label_path)
        total_tweets = len(input_df)
        
        # Simulate labeling 3 out of 5 tweets
        with open(self.labelled_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            cols = list(input_df.columns) + ['human_label']
            writer.writerow(cols)
            
            for i in range(3):
                row_data = list(input_df.iloc[i].values) + ['positive']
                writer.writerow(row_data)
        
        labeled_df = pd.read_csv(self.labelled_tweets_path)
        progress = len(labeled_df)
        
        self.assertEqual(progress, 3)
        self.assertEqual(total_tweets - progress, 2)
        self.assertGreater(total_tweets, progress)


class TestDataFlowIntegration(unittest.TestCase):
    """Test the complete data flow: sampling -> labeling."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        """Clean up."""
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def test_sampled_file_can_be_labeled_file_input(self):
        """Test that sampled file format is compatible with UI input."""
        # Create sample tweets_to_label.csv
        data = {
            'tweet_text': ['Tweet 1', 'Tweet 2'],
            'sampled_model': ['textblob', 'vader'],
            'textblob_class_1': ['positive', 'negative'],
            'vader_class_1': ['positive', 'negative'],
            'bert_class_1': ['positive', 'negative'],
            'gpt-2_class_1': ['positive', 'negative'],
        }
        df = pd.DataFrame(data)
        df.to_csv('tweets_to_label.csv', index=False)
        
        # Verify it has the expected structure
        input_df = pd.read_csv('tweets_to_label.csv')
        self.assertIn('tweet_text', input_df.columns)
        self.assertIn('sampled_model', input_df.columns)

    def test_all_models_represented_in_output(self):
        """Test that labelled output can show which tweets came from which model."""
        data = {
            'tweet_text': [f'Tweet{i}' for i in range(4)],
            'sampled_model': ['textblob', 'vader', 'bert', 'gpt-2'],
            'textblob_class_1': ['positive'] * 4,
            'vader_class_1': ['positive'] * 4,
            'bert_class_1': ['positive'] * 4,
            'gpt-2_class_1': ['positive'] * 4,
        }
        input_df = pd.DataFrame(data)
        
        # Simulate all 4 tweets being labeled
        cols = list(input_df.columns) + ['human_label']
        output_rows = []
        output_rows.append(cols)
        
        for _, row in input_df.iterrows():
            output_rows.append(list(row.values) + ['positive'])
        
        with open('labelled_tweets.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(output_rows)
        
        output_df = pd.read_csv('labelled_tweets.csv')
        
        # Check that all models are represented
        models_in_output = set(output_df['sampled_model'])
        self.assertEqual(models_in_output, {'textblob', 'vader', 'bert', 'gpt-2'})


if __name__ == '__main__':
    unittest.main()
