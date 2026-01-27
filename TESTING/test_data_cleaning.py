import unittest

class TestDataClean(unittest.TestCase):
    def test_remove_urls(self):
        text = ["Check this link: https://example.com and http://test.com"]
        self.assertEqual(
            remove_urls(text),
            ["Check this link:  and "]
        )