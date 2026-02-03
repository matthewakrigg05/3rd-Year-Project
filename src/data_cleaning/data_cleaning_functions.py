import re
import emoji


def split_camel_hashtag(s: str) -> str:
    """Split camel case hashtags into spaced words while keeping all-uppercase acronyms.

    Examples:
      BestDayEver -> Best Day Ever
      JSONData -> JSON Data
      LOL -> LOL
      worstdayever -> worstdayever
    """
    if not s:
        return s
    # Only operate on the content (no leading # expected here)
    # If the string is all lowercase or all uppercase, return as-is.
    if s.islower() or s.isupper():
        return s
    # Insert spaces on boundaries: lower->Upper OR AcronymBoundary (Upper followed by Upper+lower)
    parts = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return parts

# Common emoticon patterns -> token name
_EMOTICON_MAP = {
    r":\)": "smile",
    r":D": "laugh",
    r":\(" : "sad",
    r";\)": "wink",
    r"<3": "heart",
}
# Compile a regex to find emoticons (longest first)
_EMOTICON_REGEX = re.compile("(" + "|".join(sorted(_EMOTICON_MAP.keys(), key=len, reverse=True)) + ")")


def replace_emoticons(text: str) -> str:
    """Replace common emoticons with a token like ' EMOTICON_smile '.

    Adds surrounding spaces to make token separation safe.
    If no emoticons are found, returns the original string unchanged.
    """
    if not text:
        return text

    def _repl(m):
        key = m.group(0)
        name = _EMOTICON_MAP.get(re.escape(key), None)
        # If direct lookup fails (because keys in map are raw patterns), search mapping
        if name is None:
            for pat, nm in _EMOTICON_MAP.items():
                if re.fullmatch(pat, key):
                    name = nm
                    break
        if not name:
            return key
        return f" EMOTICON_{name} "

    out = _EMOTICON_REGEX.sub(_repl, text)
    # If nothing changed, return original (keeps exact equality for tests)
    return out

def demojize_to_tokens(text: str) -> str:
    """
    Convert emojis to tokens like 'EMOJI_face_with_tears_of_joy' or 'EMOJI_1F602'.

    If the `emoji` package is available it will use `demojize()` to get readable names
    (and strip surrounding colons). Otherwise falls back to replacing characters in
    common emoji Unicode ranges with an ordinal-based token.
    """
    if not text:
        return text

    dem = emoji.demojize(text)
    # Replace :name: with EMOJI_name (add spaces around)
    out = re.sub(r":([a-zA-Z0-9_+\-]+):", lambda m: f" EMOJI_{m.group(1)} ", dem)
    # If there were no emoji markers, return original
    if out == text:
        return text
    return out

def cap_repeated_letters(text: str) -> str:
    """Cap repeated alphabetical characters to at most three in a row."""
    if not text:
        return text
    return re.sub(r"([A-Za-z])\1{3,}", lambda m: m.group(1) * 3, text)


def cap_repeated_punct(text: str) -> str:
    """Cap repeated exclamation and question marks to at most three in a row."""
    if not text:
        return text
    return re.sub(r"([!?])\1{3,}", lambda m: m.group(1) * 3, text)

def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (spaces, tabs, newlines) to single spaces and strip ends."""
    if text is None:
        return text
    out = re.sub(r"\s+", " ", text).strip()
    return out

_URL_RE = re.compile(r"https?://\S+|www\.\S+")

def remove_urls(texts):
    """Given a list of strings, remove URLs from each string and return the new list."""
    if texts is None:
        return texts
    out = []
    for t in texts:
        if t is None:
            out.append(t)
            continue
        out.append(_URL_RE.sub("", t))
    return out

_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")

def preprocess_tweet(text: str) -> str:
    """Run lightweight preprocessing useful for downstream tasks.

    - Remove invisible characters
    - Replace URLs with 'URL'
    - Replace user mentions with 'USER'
    - Convert emoticons to tokens
    - Convert emojis to tokens
    - Convert hashtags to 'HASHTAG <text>' (split camel case when appropriate)
    - Cap repeated letters and punctuation and normalize whitespace
    """
    if not text:
        return text

    # Remove zero-width / invisible characters
    text = text.replace("\u200b", "")

    # Replace URLs with a token
    text = _URL_RE.sub("URL", text)

    # Replace user mentions with USER
    text = _USER_RE.sub("USER", text)

    # Replace emoticons
    text = replace_emoticons(text)

    # Convert emojis
    text = demojize_to_tokens(text)

    # Replace hashtags: keep case for camel-case (split words), keep lowercase as-is
    def _hashtag_repl(m):
        tag = m.group(1)
        if tag.islower():
            return f"HASHTAG {tag}"
        else:
            return f"HASHTAG {split_camel_hashtag(tag)}"

    text = _HASHTAG_RE.sub(_hashtag_repl, text)

    # Cap repeated letters and punctuation
    text = cap_repeated_letters(text)
    text = cap_repeated_punct(text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text