import pandas as pd

df = pd.read_csv('tweets_to_label.csv')
print(f'Total rows: {len(df)}')
print(f'\nColumns: {list(df.columns)[:5]}...')
print(f'\nSampled models breakdown:\n{df["sampled_model"].value_counts()}')
print(f'\nModel/Sentiment breakdown (first model):\n{df[df["sampled_model"] == "textblob"]["textblob_class_1"].value_counts()}')
print(f'\nNo duplicates check: {len(df) == len(df["tweet_text"].unique())}')
