import unittest
from src.data_collection.data_extraction import extract_english_text
from src.data_collection.data_cleaning import *

class TestExtractEnglishText(unittest.TestCase):
    def test_extracts_only_english_text(self):
        api_response = {
            "tweets": [
                {"text": "Hello world", "lang": "en"},
                {"text": "Bonjour le monde", "lang": "fr"},
                {"text": "Another English tweet", "lang": "en"},
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["Hello world", "Another English tweet"]
        )

    def test_missing_tweets_key_returns_empty_list(self):
        api_response = {"something_else": []}
        self.assertEqual(extract_english_text(api_response), [])

    def test_empty_tweets_list_returns_empty_list(self):
        api_response = {"tweets": []}
        self.assertEqual(extract_english_text(api_response), [])

    def test_ignores_items_without_text(self):
        api_response = {
            "tweets": [
                {"lang": "en"},  # no text
                {"text": "Valid", "lang": "en"},
            ]
        }
        self.assertEqual(extract_english_text(api_response), ["Valid"])

    def test_ignores_items_without_lang_or_non_en(self):
        api_response = {
            "tweets": [
                {"text": "No lang field"},               # missing lang
                {"text": "English", "lang": "en"},
                {"text": "EN uppercase", "lang": "EN"},  # not exactly "en"
            ]
        }
        self.assertEqual(extract_english_text(api_response), ["English"])

    def test_realistic_mock_shape_ignores_extra_fields(self):
        api_response = {
            "tweets": [
                {
                    "type": "tweet",
                    "id": "2010787084328714424",
                    "text": "@bananananananana She needs to be nowhere near politics.",
                    "lang": "en",
                    "author": {"userName": "tommyfcknshelby"},
                },
                {
                    "type": "tweet",
                    "id": "2",
                    "text": "Ce monde ne fera que te briser le cœur.",
                    "lang": "fr",
                    "author": {"userName": "jackcouteau"},
                },
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["@bananananananana She needs to be nowhere near politics."]
        )

    def test_extract_english_text_handles_non_list_tweets(self):
        api_response = {"tweets": "not a list"}
        self.assertEqual(extract_english_text(api_response), [])

    def test_extract_english_text_handles_missing_fields_in_tweets(self):
        api_response = {"tweets": [{}]}  # empty dict
        self.assertEqual(extract_english_text(api_response), [])

    def test_extract_english_text_with_many_tweets(self):
        tweets = [{"text": f"Tweet {i}", "lang": "en"} for i in range(1000)]
        api_response = {"tweets": tweets}
        result = extract_english_text(api_response)
        self.assertEqual(len(result), 1000)

    def test_removes_single_mention(self):
        text = ["@user Hello world"]

        self.assertEqual(
            remove_mentions(text),
            ["Hello world"]
        )

    def test_removes_multiple_mentions(self):
        text = ["@a @b @c This is a test"]
        self.assertEqual(
            remove_mentions(text),
            ["This is a test"]
        )

    def test_collapses_newlines_and_tabs(self):
        text = ["Hello\n\nworld\t\tthis   is   spaced"]
        self.assertEqual(
            collapse_whitespace(text),
            ["Hello world this is spaced"]
        )

    def test_drops_tweet_that_becomes_empty_after_cleaning_mentions(self):
        text = ["@user1 @user2"]
        self.assertEqual(
            remove_mentions(text),
            []
        )

    def test_preserves_punctuation_and_case(self):
        text = ["@user Wow!!! This works, right?"]
        self.assertEqual(
            remove_mentions(text),
            ["Wow!!! This works, right?"]
        )

    def test_removes_urls(self):
        text = ["Check this out: https://example.com @user"]
        self.assertEqual(remove_mentions(text), ["Check this out: https://example.com"])

    def test_collapse_whitespace_handles_empty_after_cleaning(self):
        text = ["   ", "\t\n"]
        self.assertEqual(collapse_whitespace(text), [])

    def test_full_cleaning_pipeline(self):
        text = ["@user   Hello\n\nworld   https://link.com"]
        cleaned = remove_mentions(text)
        cleaned = collapse_whitespace(cleaned)
        self.assertEqual(cleaned, ["Hello world https://link.com"])