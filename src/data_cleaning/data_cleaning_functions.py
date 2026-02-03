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