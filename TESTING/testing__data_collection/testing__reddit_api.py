import unittest
from unittest.mock import patch, MagicMock


class testing__reddit_client(unittest.TestCase):

    @patch("reddit_client.praw.Reddit")
    def test_successful_instance_creation(self, mock_reddit):
        client = RedditClient("id", "secret", "agent")

        mock_reddit.assert_called_once()
        self.assertIsNotNone(client.reddit)

    @patch("reddit_client.praw.Reddit")
    def test_get_subreddit_name_returns_information(self, mock_reddit):
        mock_subreddit = MagicMock()
        mock_subreddit.display_name = "Python"

        mock_reddit.return_value.subreddit.return_value = mock_subreddit

        client = RedditClient("id", "secret", "agent")
        result = client.get_subreddit_name("python")

        self.assertEqual(result, "Python")

    @patch("reddit_client.praw.Reddit")
    def test_get_hot_posts_returns_information(self, mock_reddit):
        mock_post = MagicMock()
        mock_post.title = "Test Post"

        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [mock_post]

        mock_reddit.return_value.subreddit.return_value = mock_subreddit

        client = RedditClient("id", "secret", "agent")
        posts = client.get_hot_posts("politics", limit=1)

        self.assertEqual(posts, ["Test Post"])