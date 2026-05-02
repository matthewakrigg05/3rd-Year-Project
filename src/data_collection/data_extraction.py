def extract_english_text(api_response):
    texts = []

    tweets = api_response.get("tweets", [])
    if not isinstance(tweets, list):
        return texts

    for tweet in tweets:
        if not isinstance(tweet, dict):
            continue
        if tweet.get("lang") != "en":
            continue

        text = tweet.get("text", "")

        if text:
            texts.append(text)

    return texts

