import unittest
import os
import requests
from unittest.mock import patch, MagicMock
from src.data_collection.twitter_client import TwitterClient

class TestTwitterClient(unittest.TestCase):
    def setUp(self):
        self.client = TwitterClient()

    def test_initialization(self):
        self.assertIsNotNone(self.client.url)
        self.assertIsNotNone(self.client.headers)
        self.assertIn("X-API-Key", self.client.headers)

    def test_url_format(self):
        expected_start = "https://api.twitterapi.io/twitter/tweet/advanced_search?query="
        self.assertTrue(self.client.url.startswith(expected_start))

    def test_api_key_presence(self):
        api_key = self.client.headers.get("X-API-Key")
        self.assertIsNotNone(api_key)
        self.assertNotEqual(api_key, "")

    def test_fetch_tweets_method_exists(self):
        self.assertTrue(hasattr(self.client, 'fetch_tweets'))
        self.assertTrue(callable(getattr(self.client, 'fetch_tweets')))

    @patch('requests.get')
    def test_fetch_tweets_returns_dict(self, mock_get):
        # Create a mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response
        
        response = self.client.fetch_tweets("test")
        self.assertIsInstance(response, dict)

    @patch.dict(os.environ, {"API_KEY": ""})
    def test_initialization_with_empty_api_key(self):
        with self.assertRaises(ValueError):
            TwitterClient()

    @patch('requests.get')
    def test_fetch_tweets_handles_http_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error")
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.fetch_tweets("test")

    @patch('requests.get')
    def test_fetch_tweets_handles_invalid_json(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        with self.assertRaises(ValueError):
            self.client.fetch_tweets("test")

    @patch('requests.get')
    def test_fetch_tweets_handles_rate_limit(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")
        mock_get.return_value = mock_response
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.fetch_tweets("test")

    def test_fetch_tweets_with_empty_query(self):
        # Test with empty query - should still work but might return empty results
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"tweets": []}
            mock_get.return_value = mock_response
            response = self.client.fetch_tweets("")
            self.assertIsInstance(response, dict)