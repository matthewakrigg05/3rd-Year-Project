## Project Overview

This project compares four sentiment analysis tools — TextBlob, VADER, BERT, and GPT-2 — applied to a dataset of political tweets. The goal is to evaluate each model's accuracy against manually assigned labels and to compare the computational cost of running them.

## Project Structure

- `src/` — Main scripts and supporting modules.
- `TESTING/` — Unit tests, best run via the VS Code Testing panel.

## Setup

1. Clone the repository (if appropriate):
   ```
   git clone <repository-url>
   cd 3rd-Year-Project
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in your values
   - Required: `API_KEY=your_twitter_api_key_here` (this won't work unless you have a key and credits, data from this script is stored in collected_tweets.csv)
   - Optional: `PYTHONDONTWRITEBYTECODE=1` (prevents __pycache__ creation - i find this annoying)

4. Run tests (optional):
   ```
   python -m unittest discover TESTING
   ```

## Running the Project

The scripts in `src/` should be run in the following order. All commands are run from the root of the project directory.

The pre-collected data files (`collected_tweets.csv`, `cleaned_tweets.csv`, `classified_tweets.csv`, `tweets_to_label.csv`, `labelled_tweets.csv`) are already included, so steps 1–3 can be skipped if you want to go straight to the analysis.

---

### Step 1 — Collect tweets (`collect_data.py`) *(skippable — data already provided)*

Queries the Twitter API using the political word list and saves tweets to `collected_tweets.csv`. Requires a valid API key in `.env` and can take several hours.

---

### Step 2 — Clean the dataset (`clean_data_set.py`) *(skippable — data already provided)*

Preprocesses the collected tweets (removes noise, normalises text, deduplicates) and writes the result to `cleaned_tweets.csv`.

---

### Step 3 — Classify tweets (`classify_tweets.py`) *(skippable — data already provided)*

Runs all four models against the cleaned tweets. You are prompted to choose which models to run. Results go to `classified_tweets.csv` and timing data to `model_run_time.csv`. BERT and GPT-2 are slow without a GPU.

---

### Step 4 — Sample tweets for manual labelling (`sample_tweets_for_labeling.py`) *(skippable — data already provided)*

Samples 250 tweets per sentiment class per model, with no duplicates across models. Output is saved to `tweets_to_label.csv`.

---

### Step 5 — Verify the sample (optional) (`verify_sampling.py`)

Prints a quick summary of `tweets_to_label.csv` to the console — row counts, per-model breakdown, and a duplicate check.

---

### Step 6 — Manually label tweets (`tweet_classifier_ui.py`) *(skippable — data already provided)*

Opens a Tkinter GUI that shows each tweet one at a time. Click **Positive**, **Negative**, or **Neutral** to label it. Progress is saved to `labelled_tweets.csv` and the tool resumes where you left off if reopened.

---

### Step 7 — Analyse results (`analysis.py`)

Compares model predictions against the manual labels and produces evaluation charts (accuracy, precision, recall, F1, confusion matrices, runtime comparisons). Figures are saved as numbered PNG files in `analysis_output/`.