"""
Data Collection Runner

This script runs the tweet collection pipeline.
"""

from data_collection.batch_collector import run_data_collection_pipeline

if __name__ == "__main__":
    print("Starting tweet collection pipeline...")
    print("Target: 50,000 - 60,000 tweets")
    print("This may take several hours depending on API rate limits.")
    print("Press Ctrl+C to stop early.")
    print()

    try:
        run_data_collection_pipeline(
            wordlist_file="100_political_words_phrases.txt",
            output_file="collected_tweets.csv",
            target_max=1000000
        )
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
    except Exception as e:
        print(f"Error during collection: {e}")

    print("Collection complete!")