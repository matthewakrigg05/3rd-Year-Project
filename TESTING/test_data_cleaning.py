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
        
class TestDemojizeToTokens(unittest.TestCase):
    def test_converts_known_emoji_to_token(self):
        # This requires the `emoji` package to be installed.
        out = demojize_to_tokens("fun 😂")
        self.assertIn("EMOJI_", out)

    def test_leaves_plain_text_unchanged(self):
        out = demojize_to_tokens("plain text")
        self.assertEqual(out, "plain text")

    def test_multiple_emojis(self):
        out = demojize_to_tokens("😂🔥")
        # We expect at least two emoji tokens
        self.assertGreaterEqual(out.count("EMOJI_"), 2)

    def test_emoji_token_format_has_no_colons(self):
        out = demojize_to_tokens("😂")
        self.assertNotIn(":", out)
        self.assertIn("EMOJI_", out)

class TestCapRepeatedLetters(unittest.TestCase):
    def test_caps_repeated_letters_to_three(self):
        self.assertEqual(cap_repeated_letters("sooooo"), "sooo")

    def test_does_not_change_three_or_less(self):
        self.assertEqual(cap_repeated_letters("sooo"), "sooo")
        self.assertEqual(cap_repeated_letters("soo"), "soo")

    def test_caps_multiple_runs(self):
        self.assertEqual(cap_repeated_letters("yesssss pleaaaseeee"), "yesss pleaaaseee")

    def test_only_affects_letters(self):
        self.assertEqual(cap_repeated_letters("1111!!!!"), "1111!!!!")

    def test_mixed_case_runs(self):
        # Only exact same letter repeats are capped; "AaAa" isn't a repeat run
        self.assertEqual(cap_repeated_letters("AAAAA"), "AAA")
        self.assertEqual(cap_repeated_letters("aaaaa"), "aaa")

class TestCapRepeatedPunct(unittest.TestCase):
    def test_caps_exclamation(self):
        self.assertEqual(cap_repeated_punct("wow!!!!!!"), "wow!!!")

    def test_caps_question(self):
        self.assertEqual(cap_repeated_punct("what??????"), "what???")

    def test_does_not_change_three_or_less(self):
        self.assertEqual(cap_repeated_punct("!!!"), "!!!")
        self.assertEqual(cap_repeated_punct("??"), "??")

    def test_mixed_punct_runs_not_combined(self):
        # "!?!!??" should only cap runs of same char, not merge them
        self.assertEqual(cap_repeated_punct("!?!!!!??"), "!?!!!??")

    def test_other_punct_unchanged(self):
        self.assertEqual(cap_repeated_punct("...."), "....")  # only ! and ? are handled


class TestPreprocessTweet(unittest.TestCase):
    def test_remove_urls(self):
        text = ["Check this link: https://example.com and http://test.com"]
        self.assertEqual(
            remove_urls(text),
            ["Check this link:  and "]
        )