"""Batched track selection that scales to large libraries and playlists.

The original approach sent every candidate track to the LLM in one prompt and
asked for the whole playlist back as one JSON array. That breaks on local
models: the prompt overflows the context window (Ollama silently drops the
start of it, including the instructions) and long JSON answers get cut off.

This module instead runs a tournament:

1. The candidate pool is packed into batches that fit the model's context.
2. Each batch call returns only the *numbers* of the best tracks in it
   (a few tokens per pick), so output stays tiny regardless of playlist size.
3. Winners from every batch form the next, smaller pool. Rounds repeat until
   the pool fits in a single call, which picks and orders the final playlist.
4. If the requested playlist is too large to pick in one call, the last round
   keeps exactly the requested number of tracks and the order is arranged in
   code (spreading out artists).

Picks are validated against the batch and topped up from unpicked tracks when
the model returns too few, so the result always has the requested size.
"""

import logging
import math
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from backend.llm_client import estimate_tokens
from backend.models import Track

logger = logging.getLogger(__name__)

# Share of the context window we let the prompt use (rest is safety + output)
PROMPT_BUDGET_RATIO = 0.80
# Estimated output tokens per pick: a number plus separator
TOKENS_PER_PICK = 4
# Estimated output tokens per pick when a short reason is included
TOKENS_PER_REASONED_PICK = 30
# Only ask for per-track reasons in the final call up to this many tracks
MAX_REASONED_PICKS = 60
# Never keep fewer than this share of a pool per round (quality vs. speed)
MIN_KEEP_RATIO = 0.2
# Smallest batch worth sending; below this the context is too small to be useful
MIN_BATCH_SIZE = 10
# Hard cap on batch size even for huge cloud contexts (attention degrades)
MAX_BATCH_SIZE = 400
# Retries per batch on parse failure before falling back to random picks
BATCH_ATTEMPTS = 2

FIELD_MAX_CHARS = 60


BATCH_SYSTEM = """You are a music curator choosing tracks from a numbered list for a playlist.

Pick the {k} tracks that best fit the request. Prefer variety of artists unless the request asks otherwise.

Return ONLY JSON in this exact form, using the numbers from the list:
{{"picks": [12, 3, 27]}}

Rules:
- Exactly {k} numbers, each from the list, no duplicates.
- No titles, no explanations, no markdown - just the JSON object."""


FINAL_SYSTEM = """You are a music curator building the final playlist from a numbered list of candidate tracks.

Pick exactly {k} tracks that best fit the request and put them in a good listening order (consider flow, energy and transitions). Vary artists unless the request asks otherwise.

Return ONLY JSON in this exact form, in playlist order, using the numbers from the list:
{example}

Rules:
- Exactly {k} entries, each number from the list, no duplicates.
- No markdown, no text outside the JSON object."""

FINAL_EXAMPLE_IDS = '{"picks": [12, 3, 27]}'
FINAL_EXAMPLE_REASONS = (
    '{"picks": [{"n": 12, "why": "one short sentence on why it fits"}, '
    '{"n": 3, "why": "..."}]}'
)


@dataclass
class SelectionResult:
    """Outcome of a batched selection run."""

    tracks: list[Track]
    reasons: dict[str, str] = field(default_factory=dict)  # rating_key -> reason
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    rounds: int = 0
    fallback_batches: int = 0  # batches where the LLM answer was unusable

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class _BatchOutcome:
    picks: list[int]  # indices into the batch, in model order
    reasons: dict[int, str]
    input_tokens: int = 0
    output_tokens: int = 0
    fallback: bool = False


def _clip(text: str | None) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= FIELD_MAX_CHARS else text[: FIELD_MAX_CHARS - 1] + "…"


def format_track_line(n: int, track: Track) -> str:
    """Compact one-line description of a track for the prompt."""
    parts = [f"{n}. {_clip(track.artist)} - {_clip(track.title)}"]
    meta = [_clip(track.album)] if track.album else []
    if track.year:
        meta.append(str(track.year))
    if track.genres:
        meta.append("/".join(track.genres[:2]))
    if meta:
        parts.append(f" ({', '.join(meta)})")
    return "".join(parts)


def _pack_batches(pool: list[Track], line_budget: int, max_size: int) -> list[list[Track]]:
    """Greedily split the pool into batches whose listing fits line_budget tokens."""
    batches: list[list[Track]] = []
    current: list[Track] = []
    used = 0
    for track in pool:
        cost = estimate_tokens(format_track_line(len(current) + 1, track)) + 1
        if current and (used + cost > line_budget or len(current) >= max_size):
            batches.append(current)
            current, used = [], 0
        current.append(track)
        used += cost
    if current:
        batches.append(current)
    return batches


def _allocate(total: int, sizes: list[int]) -> list[int]:
    """Split `total` picks across batches proportionally (largest remainder)."""
    pool_size = sum(sizes)
    if pool_size == 0:
        return [0] * len(sizes)
    raw = [total * s / pool_size for s in sizes]
    alloc = [min(int(r), s) for r, s in zip(raw, sizes)]
    remainder = total - sum(alloc)
    order = sorted(range(len(sizes)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    while remainder > 0:
        progressed = False
        for i in order:
            if remainder == 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def _match_by_name(item: dict, batch: list[Track], taken: set[int]) -> int | None:
    """Find a batch index for an {"artist", "title"} answer (models that ignore numbering)."""
    from rapidfuzz import fuzz

    from backend.generator import _tracks_match
    from backend.plex_client import simplify_string

    artist, title = str(item.get("artist", "")), str(item.get("title", ""))
    if not title:
        return None
    # Best match, not first match: similar titles ("Song 1" / "Song 7") both
    # clear the fuzzy threshold.
    best, best_score = None, -1.0
    for i, track in enumerate(batch):
        if i in taken or not _tracks_match(artist, title, track):
            continue
        score = fuzz.ratio(simplify_string(title), simplify_string(track.title)) + fuzz.ratio(
            simplify_string(artist), simplify_string(track.artist)
        )
        if score > best_score:
            best, best_score = i, score
    return best


def _extract_picks(data: Any, batch: list[Track]) -> tuple[list[int], dict[int, str]]:
    """Pull pick indices (and optional reasons) out of a parsed response.

    Tolerates the shapes small models actually produce: {"picks": [...]},
    a bare list, {"tracks": [...]}, numbers as strings, objects with
    n/id/number/index keys, or artist/title objects (fuzzy matched).
    """
    batch_size = len(batch)
    if isinstance(data, dict):
        for key in ("picks", "tracks", "selection", "selections", "ids", "numbers", "playlist"):
            if key in data:
                data = data[key]
                break
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if lists else []
    if not isinstance(data, list):
        return [], {}

    picks: list[int] = []
    reasons: dict[int, str] = {}
    seen: set[int] = set()
    for item in data:
        reason = ""
        value: Any = item
        if isinstance(item, dict):
            value = next(
                (item[k] for k in ("n", "id", "number", "index", "num", "track") if k in item),
                None,
            )
            reason = str(item.get("why") or item.get("reason") or "").strip()
        try:
            idx = int(str(value).strip().rstrip(".")) - 1
        except (TypeError, ValueError):
            if not isinstance(item, dict):
                continue
            matched = _match_by_name(item, batch, seen)
            if matched is None:
                continue
            idx = matched
        if 0 <= idx < batch_size and idx not in seen:
            seen.add(idx)
            picks.append(idx)
            if reason:
                reasons[idx] = reason
    return picks, reasons


class BatchedSelector:
    """Selects tracks from an arbitrarily large pool using batched LLM calls."""

    def __init__(self, llm_client: Any, request_text: str):
        self.llm = llm_client
        self.request_text = request_text.strip()
        self.context_window = llm_client.context_window
        self.concurrency = max(1, llm_client.concurrency)
        self._request_tokens = estimate_tokens(self.request_text)

    # ---- budgeting -------------------------------------------------------

    def _line_budget(self, system: str, output_tokens: int) -> int:
        """Tokens available for track listing lines in one call."""
        prompt_budget = int(self.context_window * PROMPT_BUDGET_RATIO)
        overhead = estimate_tokens(system) + self._request_tokens + 40
        return prompt_budget - overhead - output_tokens

    def _final_output_tokens(self, k: int) -> int:
        per = TOKENS_PER_REASONED_PICK if k <= MAX_REASONED_PICKS else TOKENS_PER_PICK
        return k * per + 50

    def fits_final_call(self, pool: list[Track], k: int) -> bool:
        """Whether a single call can see the whole pool and return k picks."""
        system = FINAL_SYSTEM.format(k=k, example=FINAL_EXAMPLE_REASONS)
        budget = self._line_budget(system, self._final_output_tokens(k))
        if budget <= 0 or len(pool) > MAX_BATCH_SIZE:
            return False
        listing = sum(estimate_tokens(format_track_line(i + 1, t)) + 1 for i, t in enumerate(pool))
        return listing <= budget

    def batch_capacity_estimate(self, sample: list[Track], picks_per_batch: int = 10) -> int:
        """Rough number of tracks per batch, for progress estimates."""
        system = BATCH_SYSTEM.format(k=picks_per_batch)
        budget = self._line_budget(system, picks_per_batch * TOKENS_PER_PICK + 50)
        if not sample:
            return MIN_BATCH_SIZE
        avg = sum(estimate_tokens(format_track_line(100, t)) + 1 for t in sample) / len(sample)
        return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, int(budget / avg)))

    # ---- one LLM call ----------------------------------------------------

    def _call(self, batch: list[Track], k: int, final: bool) -> _BatchOutcome:
        with_reasons = final and k <= MAX_REASONED_PICKS
        if final:
            system = FINAL_SYSTEM.format(
                k=k, example=FINAL_EXAMPLE_REASONS if with_reasons else FINAL_EXAMPLE_IDS
            )
            out_tokens = self._final_output_tokens(k)
        else:
            system = BATCH_SYSTEM.format(k=k)
            out_tokens = k * TOKENS_PER_PICK + 50

        listing = "\n".join(format_track_line(i + 1, t) for i, t in enumerate(batch))
        prompt = (
            f"{self.request_text}\n\n"
            f"Tracks (1-{len(batch)}):\n{listing}\n\n"
            f"Choose {k} of these {len(batch)} tracks. Reply with the JSON object only."
        )

        outcome = _BatchOutcome(picks=[], reasons={})
        for attempt in range(BATCH_ATTEMPTS):
            try:
                response = self.llm.generate(
                    prompt, system, max_output_tokens=out_tokens, json_mode=True
                )
                outcome.input_tokens += response.input_tokens
                outcome.output_tokens += response.output_tokens
                data = self.llm.parse_json_response(response)
                picks, reasons = _extract_picks(data, batch)
                if picks:
                    outcome.picks, outcome.reasons = picks[:k], reasons
                    return outcome
                logger.warning(
                    "Batch of %d returned no usable picks (attempt %d/%d)",
                    len(batch), attempt + 1, BATCH_ATTEMPTS,
                )
            except Exception as e:
                logger.warning(
                    "Batch of %d failed (attempt %d/%d): %s",
                    len(batch), attempt + 1, BATCH_ATTEMPTS, e,
                )
        outcome.fallback = True
        return outcome

    def _complete_picks(self, batch: list[Track], k: int, outcome: _BatchOutcome) -> list[int]:
        """Model picks, topped up from unpicked tracks so exactly k come back."""
        picks = list(outcome.picks[:k])
        if len(picks) < k:
            chosen = set(picks)
            rest = [i for i in range(len(batch)) if i not in chosen]
            random.shuffle(rest)
            picks.extend(rest[: k - len(picks)])
        return picks

    # ---- rounds ----------------------------------------------------------

    def _run_round(
        self,
        pool: list[Track],
        keep: int,
        result: SelectionResult,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[Track]:
        """Reduce `pool` to `keep` tracks via parallel batch calls."""
        # Size batches assuming a typical per-batch pick count, then allocate.
        est_ratio = keep / len(pool)
        probe = max(3, math.ceil(MAX_BATCH_SIZE * est_ratio))
        system = BATCH_SYSTEM.format(k=probe)
        line_budget = self._line_budget(system, probe * TOKENS_PER_PICK + 50)
        if line_budget < 200:
            raise ValueError(
                f"The model's context window ({self.context_window} tokens) is too small "
                "to list tracks. Use a context of at least 4096 tokens."
            )
        batches = _pack_batches(pool, line_budget, MAX_BATCH_SIZE)
        alloc = _allocate(keep, [len(b) for b in batches])

        winners: list[list[Track]] = [[] for _ in batches]
        done = 0

        def work(i: int) -> tuple[int, _BatchOutcome | None]:
            if alloc[i] == 0:
                return i, None
            if alloc[i] >= len(batches[i]):
                return i, _BatchOutcome(picks=list(range(len(batches[i]))), reasons={})
            return i, self._call(batches[i], alloc[i], final=False)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, outcome in executor.map(work, range(len(batches))):
                done += 1
                if outcome is not None:
                    if outcome.input_tokens or outcome.output_tokens or outcome.fallback:
                        result.llm_calls += 1
                    result.input_tokens += outcome.input_tokens
                    result.output_tokens += outcome.output_tokens
                    if outcome.fallback:
                        result.fallback_batches += 1
                    winners[i] = [batches[i][j] for j in self._complete_picks(batches[i], alloc[i], outcome)]
                if on_progress:
                    on_progress(done, len(batches))

        return [t for group in winners for t in group]

    def select(
        self,
        pool: list[Track],
        count: int,
        on_round: Callable[[int, int, int], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SelectionResult:
        """Select `count` tracks from `pool`.

        Args:
            pool: Candidate tracks (any size)
            count: Number of tracks wanted
            on_round: Callback(round_number, pool_size, keep) at the start of each round
            on_progress: Callback(batches_done, batches_total) within a round
        """
        result = SelectionResult(tracks=[])
        pool = list(pool)
        random.shuffle(pool)  # avoid artist/album clustering inside batches
        count = min(count, len(pool))
        if count <= 0:
            return result

        while len(pool) > count and not self.fits_final_call(pool, count):
            keep = max(count, math.ceil(len(pool) * MIN_KEEP_RATIO))
            keep = min(keep, len(pool) - 1)
            result.rounds += 1
            if on_round:
                on_round(result.rounds, len(pool), keep)
            pool = self._run_round(pool, keep, result, on_progress)
            random.shuffle(pool)

        if len(pool) > count or self.fits_final_call(pool, count):
            # Pool fits in one call: let the model pick and order the playlist.
            result.rounds += 1
            if on_round:
                on_round(result.rounds, len(pool), count)
            outcome = self._call(pool, count, final=True)
            result.llm_calls += 1
            result.input_tokens += outcome.input_tokens
            result.output_tokens += outcome.output_tokens
            if outcome.fallback:
                result.fallback_batches += 1
            picks = self._complete_picks(pool, count, outcome)
            result.tracks = [pool[i] for i in picks]
            result.reasons = {
                pool[i].rating_key: r for i, r in outcome.reasons.items() if i in set(picks)
            }
            if on_progress:
                on_progress(1, 1)
        else:
            # Too many tracks to order in one call: arrange in code.
            result.tracks = spread_artists(pool)

        return result


def spread_artists(tracks: list[Track]) -> list[Track]:
    """Order tracks so the same artist rarely plays back to back."""
    by_artist: dict[str, list[Track]] = {}
    for t in tracks:
        by_artist.setdefault((t.artist or "").casefold(), []).append(t)
    queues = sorted(by_artist.values(), key=len, reverse=True)
    ordered: list[Track] = []
    last = None
    while any(queues):
        queues.sort(key=len, reverse=True)
        for q in queues:
            if q and (q[0].artist or "").casefold() != last:
                break
        else:
            q = next(q for q in queues if q)
        track = q.pop(0)
        ordered.append(track)
        last = (track.artist or "").casefold()
    return ordered

