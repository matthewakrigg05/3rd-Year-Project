import tkinter as tk
import pandas as pd
import os
import csv

input_path = 'tweets_to_label.csv'
output_path = 'labelled_tweets.csv'

class TweetClassifier:
   
    def __init__(self):
        self.df = pd.read_csv(input_path)
        self.labelled = set()
        
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            if 'tweet_text' in output_df.columns:
                self.labelled = set(output_df['tweet_text'])
        
        # Find remaining tweets to label
        self.remaining = [i for i in range(len(self.df)) if self.df.iloc[i]['tweet_text'] not in self.labelled]
        self.current_index = 0

        self.root = tk.Tk()
        self.root.title("Tweet Classifier")

        # Display current tweet number
        self.counter_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.counter_label.pack(pady=5)

        self.tweet_label = tk.Label(self.root, text="", wraplength=400, justify=tk.LEFT)
        self.tweet_label.pack(pady=20, padx=20)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.pos_button = tk.Button(button_frame, text="Positive", command=lambda: self.classify("positive"), bg="green", fg="white")
        self.pos_button.pack(side=tk.LEFT, padx=10)

        self.neg_button = tk.Button(button_frame, text="Negative", command=lambda: self.classify("negative"), bg="red", fg="white")
        self.neg_button.pack(side=tk.LEFT, padx=10)

        self.neu_button = tk.Button(button_frame, text="Neutral", command=lambda: self.classify("neutral"), bg="gray", fg="white")
        self.neu_button.pack(side=tk.LEFT, padx=10)

        self.next_tweet()
        self.root.mainloop()

    def next_tweet(self):
        if self.current_index >= len(self.remaining):
            self.tweet_label.config(text="All tweets have been labelled!")
            self.counter_label.config(text="Complete!")
            return
        
        idx = self.remaining[self.current_index]
        self.current_row = self.df.iloc[idx]
        self.current_tweet = self.current_row['tweet_text']
        
        progress = f"Tweet {self.current_index + 1} of {len(self.remaining)}"
        self.counter_label.config(text=progress)
        self.tweet_label.config(text=self.current_tweet)

    def classify(self, sentiment):
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not os.path.exists(output_path) or os.stat(output_path).st_size == 0:
                # Get all columns from input, plus human_label
                cols = list(self.df.columns) + ['human_label']
                writer.writerow(cols)
            
            # Write the current row plus the human label
            row_data = list(self.current_row.values) + [sentiment]
            writer.writerow(row_data)
        
        self.labelled.add(self.current_tweet)
        self.current_index += 1
        self.next_tweet()

if __name__ == "__main__":
    app = TweetClassifier()