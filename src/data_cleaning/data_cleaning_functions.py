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
