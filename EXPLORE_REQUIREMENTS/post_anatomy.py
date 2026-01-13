# spike_post_anatomy.py
import praw

reddit = praw.Reddit(
    client_id="YOUR_ID",
    client_secret="YOUR_SECRET",
    user_agent="exploration_script"
)

subreddit = reddit.subreddit("politics")

for post in subreddit.hot(limit=5):
    print("=" * 80)
    print("TITLE:", repr(post.title))
    print("SELFTEXT:", repr(post.selftext[:200]))
    print("IS_SELF:", post.is_self)
    print("SCORE:", post.score)
    print("AUTHOR:", post.author)
    print("URL:", post.url)
