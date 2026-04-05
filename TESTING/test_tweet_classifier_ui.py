import unittest
import unittest.mock as mock
import pandas as pd
import os
import tempfile
import csv
from tweet_classifier_ui import TweetClassifier

class TestTweetClassifier(unittest.TestCase):

    def setUp(self):
        # Create temporary files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cleaned_tweets_path = os.path.join(self.temp_dir.name, 'cleaned_tweets.csv')
        self.classified_tweets_path = os.path.join(self.temp_dir.name, 'classified_tweets.csv')

        # Create sample cleaned_tweets.csv
        sample_data = {
            'cleaned_text': ['Tweet 1', 'Tweet 2', 'Tweet 3']
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(self.cleaned_tweets_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Tk')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Label')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Frame')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Button')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.random.shuffle')
    def test_init_no_classified_file(self, mock_shuffle, mock_button, mock_frame, mock_label, mock_tk):
        # Mock tkinter components
        mock_root = mock.MagicMock()
        mock_tk.return_value = mock_root
        mock_root.mainloop = mock.MagicMock()
        mock_tweet_label = mock.MagicMock()
        mock_label.return_value = mock_tweet_label

        # Change to temp directory for file paths
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        try:
            # Create classifier
            classifier = TweetClassifier()
            mock_shuffle.assert_called_once()
            self.assertEqual(len(classifier.remaining), 3)
            self.assertEqual(classifier.current_index, 0)

        finally:
            os.chdir(original_cwd)

    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Tk')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Label')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Frame')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.tk.Button')
    @mock.patch('src.tweet_classifier_ui.tweet_classifier_ui.random.shuffle')
    def test_init_with_classified_file(self, mock_shuffle, mock_button, mock_frame, mock_label, mock_tk):
        # Create classified_tweets.csv
        with open(self.classified_tweets_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['cleaned_text', 'classification'])
            writer.writerow(['Tweet 1', 'positive'])

        mock_root = mock.MagicMock()
        mock_tk.return_value = mock_root
        mock_root.mainloop = mock.MagicMock()
        mock_tweet_label = mock.MagicMock()
        mock_label.return_value = mock_tweet_label

        original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        try:
            classifier = TweetClassifier()
            mock_shuffle.assert_called_once()
            self.assertEqual(len(classifier.remaining), 2)  # Tweet 2 and Tweet 3

        finally:
            os.chdir(original_cwd)

    def test_next_tweet_has_tweets(self):
        classifier = TweetClassifier.__new__(TweetClassifier)
        classifier.df = pd.DataFrame({'cleaned_text': ['Tweet 1', 'Tweet 2']})
        classifier.remaining = [0, 1]
        classifier.current_index = 0
        classifier.tweet_label = mock.MagicMock()

        classifier.next_tweet()
        classifier.tweet_label.config.assert_called_with(text='Tweet 1')

    def test_next_tweet_no_more_tweets(self):
        classifier = TweetClassifier.__new__(TweetClassifier)
        classifier.df = pd.DataFrame({'cleaned_text': ['Tweet 1']})
        classifier.remaining = [0]
        classifier.current_index = 1  # Beyond length
        classifier.tweet_label = mock.MagicMock()

        classifier.next_tweet()
        classifier.tweet_label.config.assert_called_with(text="All tweets have been classified!")

    def test_classify(self):
        classifier = TweetClassifier.__new__(TweetClassifier)
        classifier.df = pd.DataFrame({'cleaned_text': ['Test Tweet', 'Next Tweet']})
        classifier.current_tweet = 'Test Tweet'
        classifier.classified = set()
        classifier.current_index = 0
        classifier.remaining = [0, 1]
        classifier.tweet_label = mock.MagicMock()

        original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        try:
            classifier.classify('positive')

            # Check file was created and has correct content
            self.assertTrue(os.path.exists('classified_tweets.csv'))
            with open('classified_tweets.csv', 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                self.assertEqual(rows[0], ['cleaned_text', 'classification'])
                self.assertEqual(rows[1], ['Test Tweet', 'positive'])

            # Check classified set updated
            self.assertIn('Test Tweet', classifier.classified)
            # Check index incremented
            self.assertEqual(classifier.current_index, 1)

        finally:
            os.chdir(original_cwd)

if __name__ == '__main__':
    unittest.main()