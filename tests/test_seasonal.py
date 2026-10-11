"""Tests for seasonal (holiday) song exclusion."""

import json
import re

import pytest

from backend.curation import BatchedSelector
from backend.llm_client import LLMClient, LLMResponse
from backend.models import LLMConfig, Track
from backend.seasonal import is_seasonal, request_is_seasonal


def t(title, album="", genres=(), key="1", artist="x"):
    return Track(rating_key=key, title=title, artist=artist, album=album,
                 duration_ms=1, genres=list(genres))


@pytest.mark.parametrize("track", [
    t("Last Christmas"), t("Happy Xmas (War Is Over)"), t("Fairytale of New York"),
    t("Sleigh Ride"), t("Feliz Navidad"), t("Run Rudolph Run"), t("Santa Baby"),
    t("Let It Snow! Let It Snow! Let It Snow!"), t("The First Noel"), t("Monster Mash"),
    t("Blue", "A Very Special Christmas"), t("Track 3", "Holiday Songs and Lullabies"),
    t("Song", genres=["Holiday"]), t("Song", genres=["Christmas"]),
    t("Silver Bells"), t("Stop the Cavalry"), t("In the Bleak Midwinter"),
    t("Mary's Boy Child"), t("O Come, O Come, Emmanuel"),
])
def test_seasonal_detected(track):
    assert is_seasonal(track)


@pytest.mark.parametrize("track", [
    t("Holiday", "Madonna"), t("Holiday", "American Idiot"), t("Snow (Hey Oh)"),
    t("Thriller"), t("Santa Monica"), t("Wonderwall", "Noel Gallagher's High Flying Birds"),
    t("Hallelujah"), t("Holidays in the Sun"), t("December"), t("Carol", "Deja Entendu"),
    t("Cold as Ice"), t("Winter Song"),
])
def test_ordinary_songs_not_flagged(track):
    assert not is_seasonal(track)


def test_request_is_seasonal():
    assert request_is_seasonal("christmas party mix")
    assert request_is_seasonal("songs for the holidays")
    assert not request_is_seasonal("rainy day jazz")
    assert not request_is_seasonal(None)


class ScreeningModel:
    """Picks the first k tracks; flags 'Fairytale' as seasonal when screening."""

    def __init__(self):
        self.screen_calls = 0
        self.rule_seen = False

    def __call__(self, prompt, system, model, max_output_tokens=None, json_mode=False):
        lines = re.findall(r"^(\d+)\. (.*)$", prompt.split("Tracks (", 1)[1], re.MULTILINE)
        if "holiday or seasonal songs?" in prompt:
            self.screen_calls += 1
            picks = [int(n) for n, text in lines if "Fairytale" in text]
        else:
            self.rule_seen = self.rule_seen or "Never pick holiday" in prompt
            k = int(re.search(r"Choose (\d+) of these", prompt).group(1))
            # Deliberately pick the hidden seasonal songs first
            seasonal = [int(n) for n, text in lines if "Fairytale" in text]
            picks = (seasonal + [int(n) for n, _ in lines if int(n) not in seasonal])[:k]
        return LLMResponse(content=json.dumps({"picks": picks}), input_tokens=10,
                           output_tokens=5, model=model)


def make_selector(exclude):
    client = LLMClient(LLMConfig(provider="ollama", model_analysis="m", model_generation="m",
                                 ollama_context_window=8192))
    fake = ScreeningModel()
    client._complete = fake
    return BatchedSelector(client, "chill", exclude_seasonal=exclude), fake


def pool():
    tracks = [t(f"Song {i}", key=str(i), artist=f"A{i}") for i in range(200)]
    tracks += [t(f"Fairytale {i}", key=f"f{i}", artist=f"F{i}") for i in range(3)]
    return tracks


def test_ai_screen_replaces_seasonal_songs_keywords_miss():
    selector, fake = make_selector(exclude=True)
    result = selector.select(pool(), 20)
    assert fake.rule_seen
    assert fake.screen_calls >= 1
    assert len(result.tracks) == 20
    assert not any("Fairytale" in x.title for x in result.tracks)
    assert result.seasonal_removed == 3
    assert len({x.rating_key for x in result.tracks}) == 20


def test_no_screen_when_disabled():
    selector, fake = make_selector(exclude=False)
    result = selector.select(pool(), 20)
    assert fake.screen_calls == 0 and not fake.rule_seen
    assert sum("Fairytale" in x.title for x in result.tracks) == 3
