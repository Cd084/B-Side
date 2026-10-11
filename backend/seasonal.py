"""Detect holiday/seasonal tracks (Christmas, Halloween, etc.).

Two layers: this keyword filter runs before the AI sees anything, and the AI
is told to skip seasonal songs and screens the finished playlist (see
curation.BatchedSelector) for ones keywords can't catch, like
"Fairytale of New York" or "Sleigh Ride".

Patterns avoid broad words that appear in ordinary songs ("Holiday" by
Madonna or Green Day, "Snow" by Red Hot Chili Peppers, "Thriller").
"""

import re

from backend.models import Track

# Unambiguous in a title or album name
_KEYWORDS = [
    r"christmas", r"x-?mas", r"navidad", r"yule(?:tide)?",
    r"santa(?! (?:monica|fe|cruz|barbara|clara|rosa|ana|maria|cecilia|catalina))",
    r"st\.? nick", r"saint nick", r"rudolph", r"reindeer",
    r"sleigh", r"jingle bells?", r"mistletoe", r"snow ?man", r"nutcracker",
    r"hanukk?ah", r"chanukk?ah", r"kwanzaa", r"halloween", r"thanksgiving",
    r"auld lang syne",
]

# Well-known seasonal titles without any of the keywords above
_TITLES = [
    r"(?:first|the) no[eë]l", r"silent night", r"holy night", r"deck the halls?", r"joy to the world",
    r"let it snow", r"winter wonderland", r"little drummer boy",
    r"hark,? the herald", r"come,? all ye faithful", r"adeste fideles",
    r"away in a manger", r"carol of the bells", r"we three kings",
    r"little town of bethlehem", r"god rest ye", r"angels we have heard",
    r"twelve days of", r"holly jolly", r"have yourself a merry",
    r"it'?s beginning to look a lot like", r"most wonderful time of the year",
    r"baby,? it'?s cold outside", r"fairytale of new york", r"feliz navidad",
    r"we wish you a merry", r"good king wenceslas",
    r"o tannenbaum", r"o christmas tree", r"monster mash", r"linus and lucy",
    r"santa baby", r"here comes santa", r"wonderful christmastime",
    r"silver bells", r"do you hear what i hear", r"mary'?s boy child",
    r"stop the cavalry", r"mele kalikimaka", r"bleak midwinter",
    r"ding,? dong,? merrily", r"o come,? o come,? emmanuel",
]

# Album names that mark a whole holiday record
_ALBUMS = [
    r"holiday (?:album|songs|collection|classics|favorites|hits|music|cheer)",
    r"(?:happy|for the|this|merry|home for the) holidays",
    r"(?:winter|holiday) (?:songs|carols)", r"carols",
]

# Genre tags Plex/MusicBrainz use for seasonal music
_GENRES = {"christmas", "holiday", "christmas music", "holiday music", "xmas", "halloween"}

_TEXT_RE = re.compile(r"\b(?:" + "|".join(_KEYWORDS + _TITLES) + r")\b", re.IGNORECASE)
_ALBUM_RE = re.compile(r"\b(?:" + "|".join(_KEYWORDS + _ALBUMS) + r")\b", re.IGNORECASE)

# The rule given to the AI in every selection call
AI_RULE = (
    "Never pick holiday or seasonal songs (Christmas, Hanukkah, Halloween, "
    "Thanksgiving, New Year's, Easter), even if they otherwise fit - including "
    "ones without the holiday in the title, like 'Fairytale of New York' or 'Sleigh Ride'."
)


def is_seasonal(track: Track) -> bool:
    """Whether a track looks seasonal from its title, album, or genre tags."""
    if _TEXT_RE.search(track.title or ""):
        return True
    if _ALBUM_RE.search(track.album or ""):
        return True
    return any((g or "").strip().lower() in _GENRES for g in track.genres or [])


def request_is_seasonal(text: str | None) -> bool:
    """Whether the user is explicitly asking for seasonal music."""
    if not text:
        return False
    return bool(_TEXT_RE.search(text) or re.search(r"\bholidays?\b", text, re.IGNORECASE))
