import unittest
from src.data_collection.data_cleaning import *

class TestCleanText(unittest.TestCase):
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