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

class TestReplaceEmoticons(unittest.TestCase):
    def test_replaces_basic_smile(self):
        self.assertIn("EMOTICON_smile", replace_emoticons("hi :)"))

    def test_replaces_basic_sad(self):
        self.assertIn("EMOTICON_sad", replace_emoticons("oh no :("))

    def test_replaces_wink(self):
        self.assertIn("EMOTICON_wink", replace_emoticons("ok ;)"))

    def test_replaces_heart(self):
        self.assertIn("EMOTICON_heart", replace_emoticons("love <3"))

    def test_replaces_laugh(self):
        self.assertIn("EMOTICON_laugh", replace_emoticons("haha :D"))

    def test_does_not_change_text_without_emoticons(self):
        inp = "no emoticons here"
        self.assertEqual(replace_emoticons(inp), inp)    

    def test_multiple_emoticons_all_replaced(self):
        out = replace_emoticons("yay :) <3 :(")
        self.assertIn("EMOTICON_smile", out)
        self.assertIn("EMOTICON_heart", out)
        self.assertIn("EMOTICON_sad", out)

    def test_spacing_added_around_tokens(self):
        # Your function returns surrounding spaces for safety.
        out = replace_emoticons("hi:)")
        self.assertIn(" EMOTICON_smile ", out)


class TestPreprocessTweet(unittest.TestCase):
    def test_remove_urls(self):
        text = ["Check this link: https://example.com and http://test.com"]
        self.assertEqual(
            remove_urls(text),
            ["Check this link:  and "]
        )