def extract_english_text(api_response):
    texts = []

    for tweet in api_response.get("tweets", []):
        if tweet.get("lang") != "en":
            continue

        text = tweet.get("text", "")

        if text:
            texts.append(text)

    return texts

