import unittest
from unittest.mock import patch, Mock
from src.data_collection import twitter_client as tc


class TestParseTweets(unittest.TestCase):
    def test_parse_tweets_happy_path(self):
        payload = {
            "data": [
                {"id": "1", "text": "hello", "author_id": "99", "created_at": "2026-01-13T10:00:00Z"},
                {"id": 2, "text": "world"},  # id as int: should still become str
            ]
        }

        tweets = tc.parse_tweets(payload)

        self.assertEqual(len(tweets), 2)
        self.assertEqual(tweets[0].id, "1")
        self.assertEqual(tweets[0].text, "hello")
        self.assertEqual(tweets[0].author_id, "99")
        self.assertIsNone(tweets[1].author_id)
        self.assertEqual(tweets[1].id, "2")

    def test_parse_tweets_missing_data(self):
        with self.assertRaisesRegex(tc.TwitterApiError, "Missing 'data' field"):
            tc.parse_tweets({"meta": {"result_count": 0}})

    def test_parse_tweets_api_error_shape(self):
        with self.assertRaisesRegex(tc.TwitterApiError, "API error"):
            tc.parse_tweets({"error": {"message": "Invalid API key"}})

    def test_parse_tweets_data_not_list(self):
        with self.assertRaisesRegex(tc.TwitterApiError, "'data' is not a list"):
            tc.parse_tweets({"data": {"id": "1", "text": "nope"}})

    def test_parse_tweets_item_not_object(self):
        with self.assertRaisesRegex(tc.TwitterApiError, "Tweet item is not an object"):
            tc.parse_tweets({"data": ["not-a-dict"]})

    def test_parse_tweets_missing_required_fields(self):
        with self.assertRaisesRegex(tc.TwitterApiError, "Tweet missing required fields"):
            tc.parse_tweets({"data": [{"id": "1"}]})


class TestFetchTweets(unittest.TestCase):
    @patch("twitter_client.requests.get")
    def test_fetch_tweets_success_mocks_http(self, mock_get):
        # Build a fake Response-like object
        fake_resp = Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"data": [{"id": "123", "text": "mocked", "lang": "en"}]}
        mock_get.return_value = fake_resp

        tweets = tc.fetch_tweets(query="python", base_url="https://example.com")

        self.assertEqual(len(tweets), 1)
        self.assertEqual(tweets[0].id, "123")
        self.assertEqual(tweets[0].text, "mocked")

        # Assert the HTTP call was made how we expect
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "https://example.com/search")
        self.assertEqual(kwargs["params"], {"q": "python"})
        self.assertEqual(kwargs["timeout"], 10.0)

    @patch("twitter_client.requests.get")
    def test_fetch_tweets_http_error_raises(self, mock_get):
        fake_resp = Mock()
        fake_resp.status_code = 401
        fake_resp.json.return_value = {"error": {"message": "unauthorized"}}
        mock_get.return_value = fake_resp

        with self.assertRaisesRegex(tc.TwitterApiError, r"HTTP 401"):
            tc.fetch_tweets(query="x", api_key="bad")