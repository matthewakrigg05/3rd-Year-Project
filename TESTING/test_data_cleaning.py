import unittest

class TestSplitCamelHashtags(unittest.TestCase):
    def test_splits_camel_case(self):
        self.assertEqual(split_camel_hashtag("BestDayEver"), "Best Day Ever")

    def test_handles_acronym_boundary(self):
        # "JSONData" should become "JSON Data"
        self.assertEqual(split_camel_hashtag("JSONData"), "JSON Data")

    def test_all_upper_kept_as_one_token(self):
        self.assertEqual(split_camel_hashtag("LOL"), "LOL")

    def test_all_lower_kept_as_one_token(self):
        self.assertEqual(split_camel_hashtag("worstdayever"), "worstdayever")

    def test_single_word_unchanged(self):
        self.assertEqual(split_camel_hashtag("Hello"), "Hello")

    


class TestPreprocessTweet(unittest.TestCase):
    def test_remove_urls(self):
        text = ["Check this link: https://example.com and http://test.com"]
        self.assertEqual(
            remove_urls(text),
            ["Check this link:  and "]
        )