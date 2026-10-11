"""Tests for batched selection over large pools with small context windows."""

import json
import re

import pytest

from backend.curation import (
    BatchedSelector,
    _allocate,
    _extract_picks,
    spread_artists,
)
from backend.llm_client import LLMClient, LLMResponse, estimate_tokens
from backend.models import LLMConfig, Track

LINE_RE = re.compile(r"^(\d+)\. (.*)$", re.MULTILINE)


def make_pool(n: int) -> list[Track]:
    """n tracks; every 10th is Jazz (the 'preferred' genre), 200 distinct artists."""
    return [
        Track(
            rating_key=str(i),
            title=f"Song Number {i} With A Moderately Long Title",
            artist=f"Artist {i % 200}",
            album=f"Album {i // 12}",
            duration_ms=200_000,
            year=1960 + i % 60,
            genres=["Jazz"] if i % 10 == 0 else ["Rock", "Pop"],
        )
        for i in range(n)
    ]


class FakeLocalModel:
    """Stands in for _complete: checks budgets, prefers Jazz, answers with numbers."""

    def __init__(self, context: int, mode: str = "numbers"):
        self.context = context
        self.mode = mode
        self.calls = 0
        self.max_prompt_tokens = 0

    def __call__(self, prompt, system, model, max_output_tokens=None, json_mode=False):
        self.calls += 1
        used = estimate_tokens(prompt) + estimate_tokens(system)
        self.max_prompt_tokens = max(self.max_prompt_tokens, used)
        assert used + (max_output_tokens or 0) <= self.context, "prompt + output overflow context"
        assert json_mode

        k = int(re.search(r"Choose (\d+) of these", prompt).group(1))
        lines = LINE_RE.findall(prompt.split("Tracks (", 1)[1])
        jazz = [int(n) for n, text in lines if "Jazz" in text]
        other = [int(n) for n, text in lines if "Jazz" not in text]
        picks = (jazz + other)[:k]

        if self.mode == "garbage":
            content = "I'm sorry, I can't help with that."
        elif self.mode == "few":
            content = json.dumps({"picks": picks[: max(1, k // 3)]})
        elif self.mode == "reasons":
            content = json.dumps({"picks": [{"n": n, "why": "fits"} for n in picks]})
        else:
            content = json.dumps({"picks": picks})
        return LLMResponse(content=content, input_tokens=used, output_tokens=k * 3, model=model)


def make_client(context: int, mode: str = "numbers") -> tuple[LLMClient, FakeLocalModel]:
    client = LLMClient(LLMConfig(
        provider="ollama",
        model_analysis="llama3:8b",
        model_generation="llama3:8b",
        ollama_context_window=context,
    ))
    fake = FakeLocalModel(context, mode)
    client._complete = fake
    return client, fake


@pytest.mark.parametrize("context,pool_size,count", [
    (4096, 3000, 25),
    (8192, 10000, 25),
    (8192, 10000, 300),   # too many to order in one call -> proportional final round
    (16384, 2000, 100),
])
def test_large_pool_fits_context_and_returns_exact_count(context, pool_size, count):
    client, fake = make_client(context)
    pool = make_pool(pool_size)

    result = BatchedSelector(client, "User's request: smoky late-night jazz").select(pool, count)

    assert len(result.tracks) == count
    keys = [t.rating_key for t in result.tracks]
    assert len(set(keys)) == count
    assert set(keys) <= {t.rating_key for t in pool}
    assert result.fallback_batches == 0
    assert fake.max_prompt_tokens < context
    assert result.llm_calls == fake.calls


def test_preference_survives_tournament():
    """The model prefers Jazz (10% of pool); the final picks should be all Jazz."""
    client, _ = make_client(8192)
    result = BatchedSelector(client, "jazz").select(make_pool(5000), 50)
    assert all("Jazz" in t.genres for t in result.tracks)


def test_small_pool_single_call_with_reasons():
    client, fake = make_client(16384, mode="reasons")
    pool = make_pool(80)
    result = BatchedSelector(client, "jazz").select(pool, 8)
    assert fake.calls == 1
    assert len(result.tracks) == 8
    assert len(result.reasons) == 8


def test_model_returning_too_few_is_topped_up():
    client, _ = make_client(8192, mode="few")
    result = BatchedSelector(client, "jazz").select(make_pool(3000), 40)
    assert len(result.tracks) == 40
    assert len({t.rating_key for t in result.tracks}) == 40


def test_garbage_responses_fall_back_without_crashing():
    client, _ = make_client(8192, mode="garbage")
    result = BatchedSelector(client, "jazz").select(make_pool(500), 20)
    assert len(result.tracks) == 20
    assert result.fallback_batches == result.llm_calls > 0


def test_count_larger_than_pool():
    client, _ = make_client(8192)
    result = BatchedSelector(client, "jazz").select(make_pool(12), 50)
    assert len(result.tracks) == 12


def test_context_too_small_raises():
    client, _ = make_client(1024)
    with pytest.raises(ValueError, match="too small"):
        BatchedSelector(client, "jazz").select(make_pool(2000), 25)


class TestExtractPicks:
    batch = make_pool(10)

    def test_object_with_picks(self):
        assert _extract_picks({"picks": [3, 1]}, self.batch)[0] == [2, 0]

    def test_bare_list_strings_and_dupes_and_out_of_range(self):
        assert _extract_picks(["2", "2.", 99, 0, "x", 5], self.batch)[0] == [1, 4]

    def test_objects_with_alternate_keys(self):
        picks, reasons = _extract_picks(
            {"tracks": [{"id": 4, "reason": "good"}, {"number": 6}]}, self.batch
        )
        assert picks == [3, 5]
        assert reasons == {3: "good"}

    def test_artist_title_objects_are_fuzzy_matched(self):
        t = self.batch[7]
        picks, _ = _extract_picks([{"artist": t.artist, "title": t.title}], self.batch)
        assert picks == [7]


def test_allocate_is_proportional_and_exact():
    alloc = _allocate(25, [150, 150, 150, 37])
    assert sum(alloc) == 25
    assert alloc[3] <= alloc[0]
    assert _allocate(10, [3, 3]) == [3, 3]


def test_spread_artists_avoids_back_to_back():
    tracks = make_pool(400)[:0] + [
        Track(rating_key=str(i), title="t", artist=a, album="x", duration_ms=1)
        for i, a in enumerate(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    ]
    ordered = spread_artists(tracks)
    assert len(ordered) == 12
    assert all(ordered[i].artist != ordered[i + 1].artist for i in range(11))


def test_ollama_request_sets_num_ctx_and_json(mocker):
    """Regression: Ollama must be told the context size or it truncates prompts."""
    client = LLMClient(LLMConfig(
        provider="ollama", model_analysis="m", model_generation="m",
        ollama_context_window=131072,
    ))
    post = mocker.patch("backend.llm_client.httpx.Client")
    post.return_value.__enter__.return_value.post.return_value.json.return_value = {
        "response": '{"picks": [1]}', "prompt_eval_count": 10, "eval_count": 3,
    }
    client.generate("prompt", "system", max_output_tokens=100, json_mode=True)
    payload = post.return_value.__enter__.return_value.post.call_args.kwargs["json"]
    assert payload["options"]["num_ctx"] == 16384  # capped native 128k
    assert payload["options"]["num_predict"] == 100
    assert payload["format"] == "json"


def test_parse_json_handles_curly_quotes():
    client = LLMClient(LLMConfig(provider="ollama", model_analysis="m", model_generation="m"))
    resp = LLMResponse(content="{“picks”: [1, 2]}", input_tokens=0, output_tokens=0, model="m")
    assert client.parse_json_response(resp) == {"picks": [1, 2]}


def test_thinking_models_get_think_false(mocker):
    client = LLMClient(LLMConfig(provider="ollama", model_analysis="m", model_generation="m"))
    http = mocker.patch("backend.llm_client.httpx.Client").return_value.__enter__.return_value
    http.post.return_value.json.side_effect = [
        {"capabilities": ["completion", "thinking"]},  # /api/show
        {"response": '{"picks": [1]}', "prompt_eval_count": 5, "eval_count": 3},
        {"response": '{"picks": [2]}', "prompt_eval_count": 5, "eval_count": 3},
    ]
    client.generate("p", "s", json_mode=True)
    client.generate("p", "s", json_mode=True)
    calls = http.post.call_args_list
    assert calls[0].args[0].endswith("/api/show")
    assert len(calls) == 3  # capability lookup is cached
    assert calls[1].kwargs["json"]["think"] is False


def test_parse_json_strips_think_block():
    client = LLMClient(LLMConfig(provider="ollama", model_analysis="m", model_generation="m"))
    resp = LLMResponse(
        content='<think>pick jazz ones {"x": 1}</think>\n{"picks": [4]}',
        input_tokens=0, output_tokens=0, model="m",
    )
    assert client.parse_json_response(resp) == {"picks": [4]}
