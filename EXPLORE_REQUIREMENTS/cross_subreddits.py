import praw

reddit = praw.Reddit(
    client_id="YOUR_ID",
    client_secret="YOUR_SECRET",
    user_agent="exploration_script"
)

subreddits = ["politics", "news", "worldnews", "ukpolitics", "all"]

for name in subreddits:
    print(f"\n### r/{name}")
    subreddit = reddit.subreddit(name)
    for post in subreddit.hot(limit=3):
        text = f"{post.title} {post.selftext}".lower()
        print("-", text[:120])
