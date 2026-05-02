import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from .twitter_client import TwitterClient
from .data_extraction import extract_english_text
from .data_cleaning import remove_mentions, collapse_whitespace
from .wordlist_loader import load_wordlist

def count_csv_rows(csv_file: str) -> int:
    """Return the number of data rows in a CSV file (excluding header)."""
    if not Path(csv_file).exists():
        return 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return sum(1 for row in reader) - 1  # Subtract header

def collect_and_save(words: List[str], 
                     output_file: str, 
                     delay: float = 0.2):
    """Fetch tweets for each word and append results to the output CSV."""
    client = TwitterClient()

    
    for word in words:
        all_texts = []

        # first attempt to fetch tweets; if this fails we skip the word
        try:
            print(f"Fetching tweets for: {word}")
            response = client.fetch_tweets(word)
        except Exception as e:
            print(f"Error fetching for '{word}': {e}")
            continue

        # process the response outside of the fetch-exception handler
        texts = extract_english_text(response)
        cleaned_texts = remove_mentions(texts)
        cleaned_texts = collapse_whitespace(cleaned_texts)
        
        # Add word context to each text
        for text in cleaned_texts:
            all_texts.append({
                'word': word,
                'text': text
            })

        # attempt to write to disk; let failures propagate so callers can handle them
        append_to_csv(all_texts, output_file)

        time.sleep(delay)  # Rate limiting


def append_to_csv(data: List[Dict[str, Any]], output_file: str):
    """Append a list of {'word', 'text'} dicts to the output CSV."""
    if not data:
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = output_path.exists()
    has_header = False
    
    if file_exists:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                has_header = first_line == "word,text"
        except:
            has_header = False
    
    with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['word', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        writer.writerows(data)
    
    print(f"Appended {len(data)} tweets to {output_file}")

def run_data_collection_pipeline(wordlist_file: str = "100_political_words_phrases.txt", 
                                output_file: str = "collected_tweets.csv",
                                target_max: int = 60000):
    """
    Run the full data collection pipeline until target tweet count is reached.
    
    Args:
        wordlist_file: Path to wordlist file
        output_file: Output CSV file
        target_min: Minimum target tweet count
        target_max: Maximum target tweet count
    """
    words = load_wordlist(wordlist_file)
    
    while True:
        current_count = count_csv_rows(output_file)
        print(f"Current tweet count: {current_count}")
        
        if current_count >= target_max:
            print(f"Target reached! Collected {current_count} tweets.")
            break
        
        words_to_use = random.sample(words, len(words))
        
        print(f"Collecting batch for {len(words_to_use)} words...")
        collect_and_save(words_to_use, output_file)
        
        # Check again after collection
        new_count = count_csv_rows(output_file)
        if new_count == current_count:
            print("No new tweets collected in this batch. Stopping to avoid infinite loop.")
            break