import unittest
from src.data_cleaning.data_cleaning_functions import *

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
        
class TestDemojiseToTokens(unittest.TestCase):
    def test_converts_known_emoji_to_token(self):
        # This requires the `emoji` package to be installed.
        out = demojise_to_tokens("fun 😂")
        self.assertIn("EMOJI_", out)

    def test_leaves_plain_text_unchanged(self):
        out = demojise_to_tokens("plain text")
        self.assertEqual(out, "plain text")

    def test_multiple_emojis(self):
        out = demojise_to_tokens("😂🔥")
        # We expect at least two emoji tokens
        self.assertGreaterEqual(out.count("EMOJI_"), 2)

    def test_emoji_token_format_has_no_colons(self):
        out = demojise_to_tokens("😂")
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

class TestNormaliseWhitespace(unittest.TestCase):
    def test_collapses_spaces(self):
        self.assertEqual(normalise_whitespace("a   b"), "a b")

    def test_collapses_tabs_newlines(self):
        self.assertEqual(normalise_whitespace("a\tb\nc"), "a b c")

    def test_strips_ends(self):
        self.assertEqual(normalise_whitespace("  a b  "), "a b")

    def test_empty_string(self):
        self.assertEqual(normalise_whitespace(""), "")

    def test_only_whitespace(self):
        self.assertEqual(normalise_whitespace("   \n\t  "), "")


class TestURLRemoval(unittest.TestCase):
    def test_remove_urls(self):
        text = ["Check this link: https://example.com and http://test.com"]
        self.assertEqual(
            remove_urls(text),
            ["Check this link:  and "]
        )


class TestStripAccents(unittest.TestCase):
    def test_strip_accents(self):
        self.assertEqual(strip_accents("café"), "cafe")
        self.assertEqual(strip_accents("déjà vu"), "deja vu")

    def test_strip_accents_unchanged(self):
        self.assertEqual(strip_accents("hello world"), "hello world")


class TestPreprocessTweet(unittest.TestCase):
    def test_replaces_url_and_user(self):
        inp = "Check this https://example.com @someone"
        out = preprocess_tweet(inp)
        self.assertIn("URL", out)
        self.assertIn("USER", out)

    def test_handles_hashtag_camelcase(self):
        inp = "So good! #BestDayEver"
        out = preprocess_tweet(inp)
        self.assertIn("HASHTAG", out)
        self.assertIn("Best Day Ever", out)

    def test_handles_hashtag_lowercase(self):
        inp = "So bad #worstdayever"
        out = preprocess_tweet(inp)
        self.assertIn("HASHTAG worstdayever", out)

    def test_emoticons_converted(self):
        inp = "great :)"
        out = preprocess_tweet(inp)
        self.assertIn("EMOTICON_smile", out)

    def test_emojis_converted(self):
        inp = "fun 😂"
        out = preprocess_tweet(inp)
        self.assertIn("EMOJI_", out)

    def test_caps_repetition(self):
        inp = "sooooo goooood!!!!!!"
        out = preprocess_tweet(inp)
        self.assertIn("sooo", out)
        self.assertIn("goood", out)  # capped to 3 o's
        self.assertIn("!!!", out)
        self.assertNotIn("!!!!!!", out)

    def test_preserves_case(self):
        inp = "I LOVE THIS"
        out = preprocess_tweet(inp)
        self.assertIn("LOVE", out)

    def test_removes_invisible_chars(self):
        inp = "he\u200bllo"
        out = preprocess_tweet(inp)
        self.assertEqual(out, "hello")

    def test_whitespace_is_normalised(self):
        inp = "a   b\tc\n\n@x  https://t.co/x"
        out = preprocess_tweet(inp)
        self.assertEqual(out.count("  "), 0)  # no double spaces
        self.assertIn("USER", out)
        self.assertIn("URL", out)

    def test_accent_normalization(self):
        inp = "café déjà vu"
        out = preprocess_tweet(inp)
        self.assertIn("cafe", out)
        self.assertIn("deja", out)
        self.assertIn("vu", out)
        self.assertNotIn("é", out)
        self.assertNotIn("à", out)

    def test_end_to_end_example(self):
        inp = "RT @John: I LOOOOVE this movie 😂😂!!! https://t.co/xyz #BestDayEver"
        out = preprocess_tweet(inp)
        self.assertIn("USER", out)
        self.assertIn("LOOOVE", out)  # capped from LOOOOVE? (already 3 O's) stays
        self.assertIn("EMOJI_", out)
        self.assertIn("!!!", out)
        self.assertIn("URL", out)
        self.assertIn("HASHTAG Best Day Ever", out)