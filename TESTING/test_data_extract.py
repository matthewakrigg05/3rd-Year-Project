import unittest
from src.data_collection.data_handling import extract_english_text

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
            ["She needs to be nowhere near politics."]
        )
    
    def test_removes_single_mention(self):
        api_response = {
            "tweets": [
                {"text": "@user Hello world", "lang": "en"}
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["Hello world"]
        )

    def test_removes_multiple_mentions(self):
        api_response = {
            "tweets": [
                {"text": "@a @b @c This is a test", "lang": "en"}
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["This is a test"]
        )

    def test_collapses_newlines_and_tabs(self):
        api_response = {
            "tweets": [
                {
                    "text": "Hello\n\nworld\t\tthis   is   spaced",
                    "lang": "en"
                }
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["Hello world this is spaced"]
        )

    def test_drops_tweet_that_becomes_empty_after_cleaning(self):
        api_response = {
            "tweets": [
                {"text": "@user1 @user2", "lang": "en"}
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            []
        )

    def test_preserves_punctuation_and_case(self):
        api_response = {
            "tweets": [
                {
                    "text": "@user Wow!!! This works, right?",
                    "lang": "en"
                }
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["Wow!!! This works, right?"]
        )

    def test_ignores_non_english_even_if_text_is_clean(self):
        api_response = {
            "tweets": [
                {"text": "@user Bonjour le monde", "lang": "fr"},
                {"text": "@user Hello world", "lang": "en"},
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["Hello world"]
        )

    def test_handles_realistic_long_reply(self):
        api_response = {
            "tweets": [
                {
                    "text": "@a @b Well I didn't read it,\n\nwhich is why I asked the question.",
                    "lang": "en"
                }
            ]
        }
        self.assertEqual(
            extract_english_text(api_response),
            ["Well I didn't read it, which is why I asked the question."]
        )
