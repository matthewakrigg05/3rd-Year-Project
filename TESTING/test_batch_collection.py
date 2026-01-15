import unittest
import tempfile
import os
import csv
from unittest.mock import patch, MagicMock
from src.data_collection.batch_collector import collect_and_save, append_to_csv, count_csv_rows

class TestBatchCollector(unittest.TestCase):
    @patch('src.data_collection.twitter_client.TwitterClient.fetch_tweets')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_collect_data_for_words_saves_to_csv(self, mock_sleep, mock_fetch):
        mock_fetch.return_value = {"tweets": [{"text": "Test tweet", "lang": "en"}]}
        words = ["politics", "government"]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            collect_and_save(words, temp_file, delay=0)
            
            self.assertTrue(os.path.exists(temp_file))
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)  # One for each word
                self.assertIn("Test tweet", [row['text'] for row in rows])
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch('src.data_collection.twitter_client.TwitterClient.fetch_tweets')
    @patch('time.sleep')
    def test_collect_data_skips_failed_requests(self, mock_sleep, mock_fetch):
        import requests
        mock_fetch.side_effect = [requests.exceptions.HTTPError("404"), {"tweets": [{"text": "Good tweet", "lang": "en"}]}]
        words = ["bad_word", "good_word"]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            collect_and_save(words, temp_file, delay=0)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 1)  # Only the good one
                self.assertEqual(rows[0]['text'], "Good tweet")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_save_to_csv_with_no_data(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            append_to_csv([], temp_file)
            # File should not be created or should be empty
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    content = f.read()
                    self.assertEqual(content, "")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch('builtins.open', side_effect=IOError("Disk full"))
    def test_collect_data_handles_csv_write_error(self, mock_open):
        words = ["test"]
        with self.assertRaises(IOError):
            collect_and_save(words, "output.csv")

    def test_count_csv_rows(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("word,text\n")
            f.write("politics,Test tweet 1\n")
            f.write("government,Test tweet 2\n")
            temp_file = f.name
        
        try:
            count = count_csv_rows(temp_file)
            self.assertEqual(count, 2)
        finally:
            os.unlink(temp_file)

    def test_count_csv_rows_nonexistent_file(self):
        count = count_csv_rows("nonexistent.csv")
        self.assertEqual(count, 0)