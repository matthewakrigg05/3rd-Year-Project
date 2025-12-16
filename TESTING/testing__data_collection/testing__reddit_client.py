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

    @patch("reddit_client.praw.Reddit")
    def test_extracts_title_and_body(self, mock_reddit):
        mock_post = MagicMock()
        mock_post.title = "Election Results"
        mock_post.selftext = "The election has concluded with..."

        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [mock_post]
        mock_reddit.return_value.subreddit.return_value = mock_subreddit

        client = RedditClient("id", "secret", "agent")
        posts = client.get_text_posts("politics", limit=1)

        self.assertEqual(posts, [
            "Election Results The election has concluded with..."
        ])

    @patch("reddit_client.praw.Reddit")
    def test_ignores_posts_without_body(self, mock_reddit):
        text_post = MagicMock()
        text_post.title = "Policy Debate"
        text_post.selftext = "Discussion on new policy"

        link_post = MagicMock()
        link_post.title = "News Link"
        link_post.selftext = ""

        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [text_post, link_post]
        mock_reddit.return_value.subreddit.return_value = mock_subreddit

        client = RedditClient("id", "secret", "agent")
        posts = client.get_text_posts("politics", limit=2)

        self.assertEqual(posts, [
            "Policy Debate Discussion on new policy"
        ])

    @patch("reddit_client.praw.Reddit")
    def test_filters_by_political_keywords(self, mock_reddit):
        political_post = MagicMock()
        political_post.title = "General Election"
        political_post.selftext = "The government announced..."

        non_political_post = MagicMock()
        non_political_post.title = "Football Match"
        non_political_post.selftext = "Match analysis"

        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [
            political_post,
            non_political_post
        ]
        mock_reddit.return_value.subreddit.return_value = mock_subreddit

        client = RedditClient("id", "secret", "agent")
        posts = client.get_political_posts("news", limit=2)

        self.assertEqual(posts, [
            "General Election The government announced..."
        ])

        @patch("reddit_client.praw.Reddit")
        def test_output_is_lowercase_and_stripped(self, mock_reddit):
            post = MagicMock()
            post.title = "Election!"
            post.selftext = "  NEW POLICY announced.  "

            mock_subreddit = MagicMock()
            mock_subreddit.hot.return_value = [post]
            mock_reddit.return_value.subreddit.return_value = mock_subreddit

            client = RedditClient("id", "secret", "agent")
            posts = client.get_clean_political_posts("politics", limit=1)

            self.assertEqual(posts, [
                "election new policy announced"
            ])