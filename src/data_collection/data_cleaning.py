import re

def remove_mentions(texts: list[str]) -> list[str]:
    clean = []

    for t in texts:
        cleaned_text = re.sub(r"@\w+", "", t)
        clean.append(cleaned_text.strip())

    clean = collapse_whitespace(clean)

    return clean

def collapse_whitespace(texts: list[str]) -> list[str]:
    cleaned = []
    for t in texts:
        cleaned_text = re.sub(r"\s+", " ", t).strip()
        if cleaned_text == "":
            continue
        else:
            cleaned.append(cleaned_text)

    return cleaned

