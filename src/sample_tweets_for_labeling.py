import pandas as pd
import random
from typing import List, Set

def sample_tweets_for_labeling():
    """
    Sample 250 tweets from each class (positive, negative, neutral) for each model.
    Ensures no duplicate tweets across all models.
    Creates tweets_to_label.csv with sampled_model column.
    """
    
    # Load the classified tweets with model predictions
    df = pd.read_csv('classified_tweets.csv')
    
    # Models and their prediction columns
    models = {
        'textblob': 'textblob_class_1',
        'vader': 'vader_class_1',
        'bert': 'bert_class_1',
        'gpt-2': 'gpt-2_class_1'
    }
    
    classes = ['positive', 'negative', 'neutral']
    samples_per_class = 250
    
    # Track all tweets already sampled to avoid duplicates
    sampled_indices: Set[int] = set()
    sampled_data = []
    
    print("Sampling tweets for labeling...")
    
    # For each model
    for model_name, pred_column in models.items():
        print(f"\nProcessing {model_name}...")
        
        # For each class
        for sentiment_class in classes:
            # Get indices of tweets predicted as this class
            class_mask = df[pred_column] == sentiment_class
            available_indices = set(df[class_mask].index) - sampled_indices
            available_indices = list(available_indices)
            
            # Sample 250 (or fewer if not enough available)
            num_to_sample = min(samples_per_class, len(available_indices))
            
            if num_to_sample < samples_per_class:
                print(f"  WARNING: Only {num_to_sample}/{samples_per_class} tweets available for {sentiment_class}")
            
            sampled_indices_for_class = random.sample(available_indices, num_to_sample)
            
            # Add these to sampled data
            for idx in sampled_indices_for_class:
                row = df.loc[idx].copy()
                row['sampled_model'] = model_name
                sampled_data.append(row)
                sampled_indices.add(idx)
            
            print(f"  Sampled {num_to_sample} {sentiment_class} tweets")
    
    # Create dataframe from sampled data
    output_df = pd.DataFrame(sampled_data)
    
    # Reorder columns to put sampled_model near the start
    cols = output_df.columns.tolist()
    cols.remove('sampled_model')
    cols.insert(1, 'sampled_model')  # Put after tweet_text
    output_df = output_df[cols]
    
    # Save to tweets_to_label.csv
    output_df.to_csv('tweets_to_label.csv', index=False, encoding='utf-8')
    
    print(f"\nSuccessfully sampled {len(sampled_data)} tweets")
    print(f"Saved to tweets_to_label.csv")
    print(f"Total tweets sampled: {len(sampled_data)}")
    print(f"Breakdown per model: {len(sampled_data) // 4} tweets per model")
    
    return output_df

if __name__ == "__main__":
    sample_tweets_for_labeling()
