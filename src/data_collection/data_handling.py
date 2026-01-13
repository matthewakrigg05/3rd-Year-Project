import re


def extract_english_text(api_response):
    texts = []

    for tweet in api_response.get("tweets", []):
        if tweet.get("lang") != "en":
            continue

        text = tweet.get("text", "")
        text = re.sub(r"@\w+", "", text)  # remove mentions
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            texts.append(text)

    return texts