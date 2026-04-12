"""
Batch sentiment classification of tweets using all available models.

This script:
1. Loads cleaned tweets from cleaned_tweets.csv
2. Allows user to select which models to run
3. Runs selected model(s) with GPU/device information
4. Saves results immediately after each model run
5. Records all scores and classifications
6. Tracks performance metrics (timing, CPU/GPU usage)
7. Outputs:
   - classified_tweets.csv: tweet classification results
   - model_run_time.csv: performance and timing data

Usage:
    python -m src.classify_tweets
"""

import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import psutil
import platform
import torch

from models.bert_method import BertSentimentAnalyser
from models.vader_method import VaderSentimentAnalyser
from models.textblob_method import TextBlobSentimentAnalyser
from models.llm_method import GPT2SentimentAnalyser
from utils.data_loading import load_csv_to_dataframe


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TweetClassifier:
    """
    Manages batch classification of tweets using multiple sentiment analysis models.
    """
    
    def __init__(self, input_csv: str = "cleaned_tweets.csv", 
                 output_classified: str = "classified_tweets.csv",
                 output_timing: str = "model_run_time.csv",
                 text_column: str = "cleaned_text"):
        """
        Initialize the classifier.
        
        Args:
            input_csv: Path to input CSV with tweets
            output_classified: Path to output CSV for classifications
            output_timing: Path to output CSV for timing/performance data
            text_column: Column name containing tweet text to classify
        """
        self.input_csv = Path(input_csv)
        self.output_classified = Path(output_classified)
        self.output_timing = Path(output_timing)
        self.text_column = text_column
        
        self.tweets_df = None
        self.models = {}
        self.results = {}
        self.timing_data = []
        self.models_to_run: Set[str] = set()  # Track which models to run
        
    def load_tweets(self) -> int:
        """
        Load tweets from CSV file.
        
        Returns:
            Number of tweets loaded
        """
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")
        
        logger.info(f"Loading tweets from {self.input_csv}...")
        self.tweets_df = load_csv_to_dataframe(self.input_csv)
        
        if self.text_column not in self.tweets_df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in CSV. Available: {self.tweets_df.columns.tolist()}")
        
        num_tweets = len(self.tweets_df)
        logger.info(f"Loaded {num_tweets} tweets")
        return num_tweets
    
    def initialize_models(self) -> None:
        """Initialize all sentiment analysis models."""
        logger.info("Initializing models...")
        
        try:
            self.models = {
                "BERT": BertSentimentAnalyser(),
                "VADER": VaderSentimentAnalyser(),
                "TextBlob": TextBlobSentimentAnalyser(),
                "GPT-2": GPT2SentimentAnalyser(),
            }
            logger.info(f"Initialized {len(self.models)} models: {', '.join(self.models.keys())}")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def get_device_info(self) -> Dict[str, str]:
        """
        Get information about the processing device (CPU/GPU) and environment.
        
        Returns:
            Dictionary with device information
        """
        device_info = {
            "platform": platform.system(),
            "cpu_count": str(psutil.cpu_count()),
            "cpu_percent": f"{psutil.cpu_percent(interval=1):.1f}%",
            "memory_available_gb": f"{psutil.virtual_memory().available / (1024**3):.2f}",
        }
        
        # Add GPU information
        if torch.cuda.is_available():
            device_info["cuda_available"] = "True"
            device_info["cuda_device"] = torch.cuda.get_device_name(0)
            device_info["gpu_memory_total_mb"] = str(torch.cuda.get_device_properties(0).total_memory / (1024**2))
        else:
            device_info["cuda_available"] = "False"
            device_info["cuda_device"] = "No GPU"
            device_info["gpu_memory_total_mb"] = "0"
            
        return device_info
    
    def classify_batch(self, model_name: str, model, 
                      run_number: int) -> Tuple[Dict[int, Tuple[float, str]], float, Dict]:
        """
        Classify all tweets using a single model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            run_number: Which run number (1-3)
            
        Returns:
            Tuple of (results_dict, elapsed_time, device_info)
            where results_dict maps tweet_index -> (score, label)
        """
        device_info = self.get_device_info()
        
        # Print device information before running the model
        logger.info(f"\n{'='*70}")
        logger.info(f"Running {model_name} (run {run_number}/3)")
        logger.info(f"{'='*70}")
        logger.info(f"Device Configuration:")
        logger.info(f"  Platform: {device_info.get('platform', 'Unknown')}")
        logger.info(f"  CPU Count: {device_info.get('cpu_count', 'Unknown')}")
        logger.info(f"  CPU Usage: {device_info.get('cpu_percent', 'Unknown')}")
        logger.info(f"  RAM Available: {device_info.get('memory_available_gb', 'Unknown')} GB")
        cuda_available = device_info.get('cuda_available', 'False')
        logger.info(f"  GPU Available: {cuda_available}")
        if cuda_available == "True":
            logger.info(f"  GPU Name: {device_info.get('cuda_device', 'Unknown')}")
            logger.info(f"  GPU Memory: {device_info.get('gpu_memory_total_mb', 'Unknown')} MB")
        logger.info(f"{'='*70}")
        logger.info(f"Processing {len(self.tweets_df)} tweets...")
        
        results_dict = {}
        start_time = time.time()
        tweets_processed = 0
        errors_encountered = 0
        
        for idx, row in self.tweets_df.iterrows():
            try:
                text = str(row[self.text_column])
                result = model.analyse(text)
                
                score = result.get("score")
                label = result.get("label")
                
                # Handle missing values
                if score is None:
                    score = None
                    label = "ERROR: missing score"
                
                results_dict[idx] = (score, label)
                tweets_processed += 1
                
            except Exception as e:
                errors_encountered += 1
                logger.warning(f"Error processing tweet {idx}: {str(e)[:100]}")
                results_dict[idx] = (None, f"ERROR: {str(e)[:50]}")
            
            # Progress indicator
            if (tweets_processed + errors_encountered) % 500 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  {tweets_processed + errors_encountered}/{len(self.tweets_df)} processed in {elapsed:.1f}s")
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n{model_name} (run {run_number}) completed in {elapsed_time:.2f}s")
        logger.info(f"  Processed: {tweets_processed}, Errors: {errors_encountered}")
        
        return results_dict, elapsed_time, device_info
    
    def run_selected_models(self) -> None:
        """
        Run only the selected models, saving results immediately after each model.
        """
        logger.info("\n" + "=" * 70)
        logger.info("Starting classification process...")
        logger.info(f"Running {len(self.models_to_run)} model(s): {', '.join(sorted(self.models_to_run))}")
        logger.info("=" * 70)
        
        # Initialize results structure for each selected model
        self.results = {model_name: {} for model_name in self.models_to_run}
        
        # Run each selected model 3 times
        for model_name in sorted(self.models_to_run):
            model = self.models[model_name]
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Model: {model_name} - Running 3 iterations")
            logger.info(f"{'=' * 70}")
            
            for run_num in range(1, 4):
                results_dict, elapsed_time, device_info = self.classify_batch(
                    model_name, model, run_num
                )
                
                self.results[model_name][run_num] = results_dict
                
                # Record timing data
                self.timing_data.append({
                    "model": model_name,
                    "run": run_num,
                    "total_time_seconds": f"{elapsed_time:.2f}",
                    "tweets_processed": len([v for v in results_dict.values() if v[0] is not None]),
                    "platform": device_info.get("platform", ""),
                    "cpu_count": device_info.get("cpu_count", ""),
                    "cpu_percent": device_info.get("cpu_percent", ""),
                    "memory_available_gb": device_info.get("memory_available_gb", ""),
                    "cuda_available": device_info.get("cuda_available", ""),
                    "cuda_device": device_info.get("cuda_device", ""),
                    "gpu_memory_total_mb": device_info.get("gpu_memory_total_mb", ""),
                })
            
            # Save results immediately after all runs of this model
            logger.info(f"\nSaving results for {model_name}...")
            self.save_classifications()
            self.save_timing_data()
            logger.info(f"Results for {model_name} saved!")
        
        logger.info(f"\n{'=' * 70}")
        logger.info("Classification complete!")
        logger.info(f"{'=' * 70}")
    
    def save_classifications(self) -> None:
        """
        Save classification results to CSV, merging with any existing results.
        Only updates/creates columns for models/runs that have just completed.
        """
        logger.info(f"\nSaving classifications to {self.output_classified} (merging with previous results if present)...")

        # Load existing results if file exists
        if self.output_classified.exists():
            existing_df = pd.read_csv(self.output_classified)
        else:
            existing_df = pd.DataFrame()

        # Build new results DataFrame for current run
        models_in_results = sorted(self.results.keys())
        new_rows = []
        for idx, row in self.tweets_df.iterrows():
            csv_row = {"tweet_text": row[self.text_column]}
            for model_name in models_in_results:
                for run_num in range(1, 4):
                    if run_num not in self.results[model_name]:
                        continue
                    score, label = self.results[model_name][run_num].get(
                        idx, (None, "ERROR: missing result")
                    )
                    score_col = f"{model_name.lower()}_score_{run_num}"
                    label_col = f"{model_name.lower()}_class_{run_num}"
                    csv_row[score_col] = score if score is not None else "null"
                    csv_row[label_col] = label if label is not None else "null"
            new_rows.append(csv_row)
        new_df = pd.DataFrame(new_rows)

        # Merge with existing results on tweet_text
        if not existing_df.empty:
            merged_df = pd.merge(existing_df, new_df, on="tweet_text", how="outer", suffixes=("", "_new"))
            # For any columns that exist in both, prefer the new (just-run) results
            for col in new_df.columns:
                if col != "tweet_text" and col in merged_df.columns and col + "_new" in merged_df.columns:
                    merged_df[col] = merged_df[col + "_new"].combine_first(merged_df[col])
                    merged_df.drop(columns=[col + "_new"], inplace=True)
        else:
            merged_df = new_df

        # Write merged results to CSV
        merged_df.to_csv(self.output_classified, index=False)
        logger.info(f"Saved {len(merged_df)} classifications to {self.output_classified}")
    
    def save_timing_data(self) -> None:
        """
        Save timing and performance data to CSV (including GPU information).
        Appends new data to existing CSV without overwriting.
        """
        logger.info(f"\nSaving timing data to {self.output_timing}...")
        
        if self.timing_data:
            fieldnames = [
                "model", "run", "total_time_seconds", "tweets_processed",
                "platform", "cpu_count", "cpu_percent", "memory_available_gb",
                "cuda_available", "cuda_device", "gpu_memory_total_mb"
            ]
            
            # Check if file exists and has data
            file_exists = self.output_timing.exists() and self.output_timing.stat().st_size > 0
            
            with open(self.output_timing, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                # Only write header if file is new or empty
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.timing_data)
        
        logger.info(f"Saved {len(self.timing_data)} timing records to {self.output_timing}")
    
    def select_models_to_run(self) -> None:
        """
        Interactively prompt user to select which models to run.
        """
        logger.info("\n" + "=" * 70)
        logger.info("MODEL SELECTION")
        logger.info("=" * 70)
        logger.info("\nAvailable models:")
        
        available_models = sorted(self.models.keys())
        for i, model_name in enumerate(available_models, 1):
            logger.info(f"  {i}. {model_name}")
        
        logger.info(f"  {len(available_models) + 1}. Run all models")
        logger.info(f"  0. Cancel")
        
        # Get user input
        while True:
            try:
                user_input = input("\nEnter model number(s) to run (comma-separated, e.g., '1,3' or 'all'): ").strip().lower()
                
                if user_input == "0":
                    logger.info("Cancelled by user.")
                    return False
                
                if user_input == "all" or user_input == str(len(available_models) + 1):
                    self.models_to_run = set(available_models)
                    logger.info(f"Selected all models: {', '.join(sorted(self.models_to_run))}")
                    break
                
                # Parse comma-separated numbers
                selected_indices = [int(x.strip()) for x in user_input.split(",")]
                self.models_to_run = set()
                
                for idx in selected_indices:
                    if 1 <= idx <= len(available_models):
                        self.models_to_run.add(available_models[idx - 1])
                    else:
                        logger.warning(f"Invalid selection: {idx}")
                
                if self.models_to_run:
                    logger.info(f"Selected models: {', '.join(sorted(self.models_to_run))}")
                    break
                else:
                    logger.warning("No valid models selected. Please try again.")
                    
            except (ValueError, IndexError):
                logger.warning("Invalid input. Please enter valid model numbers separated by commas (e.g., '1,3').")
        
        return True
    
    def run(self) -> None:
        """Execute the complete classification workflow."""
        try:
            # Load tweets and initialize models
            self.load_tweets()
            self.initialize_models()
            
            # Ask user which models to run
            if not self.select_models_to_run():
                return
            
            # Run selected models
            self.run_selected_models()
            
            logger.info("\n" + "=" * 70)
            logger.info("SUCCESS: All selected model classifications completed!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    classifier = TweetClassifier()
    classifier.run()


if __name__ == "__main__":
    main()
