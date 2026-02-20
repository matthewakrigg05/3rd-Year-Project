import csv
import unittest
from pathlib import Path


CSV_PATH = Path(__file__).resolve().parents[1] / "collected_tweets.csv"


class TestCSVLoad(unittest.TestCase):
	def test_csv_headers(self):
		with CSV_PATH.open(newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			header = next(reader)

		self.assertEqual(header, ["word", "text"])

	def test_each_row_has_two_columns_and_no_empty_fields(self):
		with CSV_PATH.open(newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			next(reader)  # skip header
			for i, row in enumerate(reader, start=1):
				self.assertEqual(len(row), 2, f"Row {i} does not have 2 columns: {row}")
				self.assertNotEqual(row[0].strip(), "", f"Row {i} has empty 'word'")
				self.assertNotEqual(row[1].strip(), "", f"Row {i} has empty 'text'")

	def test_contains_expected_category(self):
		found = False
		with CSV_PATH.open(newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			next(reader)
			for row in reader:
				if row[0].strip().lower() == "free speech":
					found = True
					break

		self.assertTrue(found, "Expected to find category 'free speech' in CSV")

	def test_row_count_nonzero(self):
		with CSV_PATH.open(newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			next(reader)  # header
			# ensure at least one data row exists
			has_row = any(True for _ in reader)

		self.assertTrue(has_row, "Expected at least one data row in CSV")


if __name__ == "__main__":
	unittest.main()

