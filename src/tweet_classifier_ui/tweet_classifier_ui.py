import tkinter as tk
import pandas as pd
import random
import os
import csv

class TweetClassifier:
    def __init__(self):
        self.df = pd.read_csv('cleaned_tweets.csv')
        self.classified = set()
        classified_path = 'classified_tweets.csv'
        if os.path.exists(classified_path):
            classified_df = pd.read_csv(classified_path)
            if 'cleaned_text' in classified_df.columns:
                self.classified = set(classified_df['cleaned_text'])
        self.remaining = [i for i in range(len(self.df)) if self.df.iloc[i]['cleaned_text'] not in self.classified]
        random.shuffle(self.remaining)
        self.current_index = 0

        self.root = tk.Tk()
        self.root.title("Tweet Classifier")

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
            self.tweet_label.config(text="All tweets have been classified!")
            return
        idx = self.remaining[self.current_index]
        self.current_tweet = self.df.iloc[idx]['cleaned_text']
        self.tweet_label.config(text=self.current_tweet)

    def classify(self, sentiment):
        classified_path = 'classified_tweets.csv'
        with open(classified_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not os.path.exists(classified_path) or os.stat(classified_path).st_size == 0:
                writer.writerow(['cleaned_text', 'classification'])
            writer.writerow([self.current_tweet, sentiment])
        self.classified.add(self.current_tweet)
        self.current_index += 1
        self.next_tweet()

if __name__ == "__main__":
    app = TweetClassifier()