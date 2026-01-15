import unittest
import tempfile
import os
from unittest.mock import patch
from src.data_collection.batch_collector import run_data_collection_pipeline

class TestIntegration(unittest.TestCase):
    @patch('src.data_collection.batch_collector.load_wordlist')
    @patch('src.data_collection.batch_collector.TwitterClient.fetch_tweets')
    @patch('src.data_collection.batch_collector.append_to_csv')
    @patch('src.data_collection.batch_collector.time.sleep')
    def test_full_pipeline_integration(self, mock_sleep, mock_append, mock_fetch, mock_load):
        mock_load.return_value = ["politics"]
        mock_fetch.return_value = {"tweets": [{"text": "Political tweet", "lang": "en"}]}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            run_data_collection_pipeline(wordlist_file="dummy.txt", output_file=temp_file)
            mock_append.assert_called_once()
            # Check that append was called with the processed data
            call_args = mock_append.call_args[0]
            data = call_args[0]
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['word'], "politics")
            self.assertEqual(data[0]['text'], "Political tweet")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)