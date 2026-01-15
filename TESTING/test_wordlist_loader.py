import unittest
import tempfile
import os
from src.data_collection.wordlist_loader import load_wordlist

class TestWordlistLoader(unittest.TestCase):
    def test_load_wordlist_returns_list(self):
        words = load_wordlist("100_political_words_phrases.txt")
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)
        self.assertIn("politics", words)  # Check if a known word is present

    def test_load_wordlist_handles_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_wordlist("nonexistent.txt")

    def test_load_wordlist_handles_empty_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', dir=os.path.dirname(__file__) + "/../") as f:
            f.write("")
            temp_file = os.path.basename(f.name)
        
        try:
            words = load_wordlist(temp_file)
            self.assertEqual(words, [])
        finally:
            os.unlink(os.path.join(os.path.dirname(__file__), "..", temp_file))