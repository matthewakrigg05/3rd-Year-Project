import csv
import unittest
from pathlib import Path
from src.data_loading import *
import pandas as pd


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

	def test_load_csv_to_dataframe_returns_dataframe(self):
		df = load_csv_to_dataframe(CSV_PATH)

		self.assertIsInstance(df, pd.DataFrame)
		self.assertListEqual(list(df.columns), ["word", "text"])
		self.assertGreaterEqual(len(df), 1)

	def test_dataframe_contains_expected_category(self):
		df = load_csv_to_dataframe(CSV_PATH)

		self.assertTrue(
			(df["word"].str.lower() == "free speech").any(),
			"Expected to find category 'free speech' in DataFrame",
		)

	def test_iter_csv_chunks_yields_dataframe_chunks(self):
		# compute expected total rows in file (excluding header)
		with CSV_PATH.open(newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			next(reader)
			expected_rows = sum(1 for _ in reader)

		# collect chunks (iterator may be lazy; materialise in test)
		chunks = list(iter_csv_chunks(CSV_PATH, chunksize=50))

		self.assertGreater(len(chunks), 0, "Expected iter_csv_chunks to yield at least one chunk")

		total = 0
		for c in chunks:
			self.assertIsInstance(c, pd.DataFrame)
			total += len(c)

		self.assertEqual(total, expected_rows)

	def test_load_dataset_without_chunksize(self):
		df = load_dataset(CSV_PATH)

		self.assertIsInstance(df, pd.DataFrame)
		self.assertListEqual(list(df.columns), ["word", "text"])

	def test_load_dataset_with_chunksize(self):
		with CSV_PATH.open(newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			next(reader)
			expected_rows = sum(1 for _ in reader)

		df = load_dataset(CSV_PATH, chunksize=25)

		self.assertIsInstance(df, pd.DataFrame)
		self.assertEqual(len(df), expected_rows)

	def test_stream_dataset_yields_chunks_and_sums(self):
		with CSV_PATH.open(newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			next(reader)
			expected_rows = sum(1 for _ in reader)

		# ensure generator yields DataFrame chunks and sums to expected rows
		first = next(stream_dataset(CSV_PATH, chunksize=50))
		self.assertIsInstance(first, pd.DataFrame)

		total = 0
		for chunk in stream_dataset(CSV_PATH, chunksize=50):
			self.assertIsInstance(chunk, pd.DataFrame)
			total += len(chunk)

		self.assertEqual(total, expected_rows)
