import unittest

def extract_english_text(api_response):
    return [
        tweet["text"]
        for tweet in api_response.get("tweets", [])
        if tweet.get("lang") == "en" and "text" in tweet
    ]


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
                    "text": "@NathanBrandWA She needs to be nowhere near politics.",
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
            ["@NathanBrandWA She needs to be nowhere near politics."]
        )