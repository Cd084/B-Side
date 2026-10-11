"""Playlist generation with library validation."""

import json
import logging
import queue
import threading
from collections.abc import Generator
from datetime import datetime

from backend.curation import BatchedSelector, SelectionResult
from backend.seasonal import is_seasonal, request_is_seasonal
from backend.llm_client import MAX_TRACK_POOL, LLMResponse, get_llm_client
from backend.models import GenerateResponse, Track
from backend.plex_client import PlexQueryError, get_plex_client
from backend import library_cache

logger = logging.getLogger(__name__)


def generate_narrative(
    track_selections: list[dict],
    llm_client,
    user_request: str = "",
) -> tuple[str, str]:
    """Generate a creative title and narrative for the playlist.

    Args:
        track_selections: List of track dicts with artist, title, album, reason
        llm_client: LLM client instance
        user_request: Original user prompt/request for context

    Returns:
        Tuple of (playlist_title with date, narrative)
        On failure, returns ("{Mon YYYY} Playlist", "")
    """
    # Build input for Query 2: track list with reasons
    tracks_with_reasons = "\n".join(
        f"- {sel.get('artist', 'Unknown')} - \"{sel.get('title', 'Unknown')}\": {sel.get('reason', 'Selected for this playlist')}"
        for sel in track_selections[:15]  # Limit to first 15 for context efficiency
    )

    # Include user request for context
    if user_request:
        narrative_prompt = f"User's request: {user_request}\n\nSelected tracks:\n{tracks_with_reasons}"
    else:
        narrative_prompt = f"Selected tracks:\n{tracks_with_reasons}"

    # Get current month/year for title suffix
    date_suffix = datetime.now().strftime("%b %Y")
    fallback_title = f"{date_suffix} Playlist"

    try:
        # Use analysis model for better creative writing quality
        response = llm_client.analyze(narrative_prompt, NARRATIVE_SYSTEM)
        result = llm_client.parse_json_response(response)

        # Handle array-wrapped responses (some LLMs wrap in [])
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if not isinstance(result, dict):
            logger.warning("Narrative response not a dict: %s", type(result).__name__)
            return fallback_title, ""

        raw_title = result.get("title", "").strip()

        # Try common alternate keys for narrative
        narrative = (
            result.get("narrative")
            or result.get("description")
            or result.get("text")
            or result.get("content")
            or ""
        ).strip()

        # Log if we got title but no narrative (helps debug)
        if raw_title and not narrative:
            logger.warning("Narrative missing from response. Keys: %s", list(result.keys()))

        # Append date to title
        if raw_title:
            playlist_title = f"{raw_title} - {date_suffix}"
        else:
            playlist_title = fallback_title

        return playlist_title, narrative

    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return fallback_title, ""


def _cached_track_to_model(cached: dict) -> Track:
    """Convert a cached track dict to a Track model."""
    return Track(
        rating_key=cached["rating_key"],
        title=cached["title"],
        artist=cached["artist"],
        album=cached["album"],
        duration_ms=cached.get("duration_ms") or 0,
        year=cached.get("year"),
        genres=cached.get("genres") or [],
        art_url=f"/api/art/{cached['rating_key']}",
    )


# The candidate pool is at least this many times the playlist size, so the AI
# always has a real choice to make (e.g. 500 tracks -> scan at least 2000).
MIN_POOL_MULTIPLIER = 4


def effective_pool_size(max_tracks_to_ai: int, track_count: int) -> int:
    """Number of candidate tracks to load for a request (0 = no limit)."""
    if max_tracks_to_ai <= 0:
        return MAX_TRACK_POOL
    return min(MAX_TRACK_POOL, max(max_tracks_to_ai, track_count * MIN_POOL_MULTIPLIER))


def _get_tracks_from_cache_or_plex(
    plex_client,
    genres: list[str] | None,
    decades: list[str] | None,
    exclude_live: bool,
    min_rating: int,
    max_tracks_to_ai: int,
) -> list[Track]:
    """Get tracks from cache if available, otherwise from Plex.

    Returns:
        List of Track objects
    """
    has_filters = genres or decades or min_rating > 0
    effective_limit = max_tracks_to_ai if max_tracks_to_ai > 0 else MAX_TRACK_POOL

    # Try cache first
    if library_cache.has_cached_tracks():
        logger.info("Using cached tracks for generation")
        cached_tracks = library_cache.get_tracks_by_filters(
            genres=genres,
            decades=decades,
            min_rating=min_rating,
            exclude_live=exclude_live,
            limit=effective_limit,
        )
        return [_cached_track_to_model(t) for t in cached_tracks]

    # Fall back to Plex
    logger.info("Cache empty, fetching from Plex")
    if not has_filters:
        return plex_client.get_random_tracks(
            count=effective_limit,
            exclude_live=exclude_live,
        )
    else:
        return plex_client.get_tracks_by_filters(
            genres=genres,
            decades=decades,
            exclude_live=exclude_live,
            min_rating=min_rating,
            limit=effective_limit,
        )


def generate_playlist_stream(
    prompt: str | None = None,
    seed_track: Track | None = None,
    selected_dimensions: list[str] | None = None,
    additional_notes: str | None = None,
    refinement_answers: list[str | None] | None = None,
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    track_count: int = 25,
    exclude_live: bool = True,
    min_rating: int = 0,
    max_tracks_to_ai: int = 500,
    exclude_seasonal: bool = True,
) -> Generator[str, None, None]:
    """Generate a playlist with streaming progress updates.

    Yields SSE-formatted events with progress updates and final result.
    """
    def emit(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        logger.info("Starting playlist generation (streaming)")
        llm_client = get_llm_client()
        plex_client = get_plex_client()

        if not llm_client:
            yield emit("error", {"message": "LLM client not initialized"})
            return
        if not plex_client:
            yield emit("error", {"message": "Plex client not initialized"})
            return

        has_filters = genres or decades or min_rating > 0

        # Step 1: Fetch tracks from cache or Plex
        using_cache = library_cache.has_cached_tracks()
        if using_cache:
            yield emit("progress", {"step": "fetching", "message": "Loading tracks from cache..."})
        elif not has_filters:
            yield emit("progress", {"step": "fetching", "message": "Sampling random tracks from library..."})
        else:
            yield emit("progress", {"step": "fetching", "message": "Fetching tracks from library..."})

        logger.info("Fetching tracks: genres=%s, decades=%s, min_rating=%s, using_cache=%s",
                    genres, decades, min_rating, using_cache)
        try:
            filtered_tracks = _get_tracks_from_cache_or_plex(
                plex_client=plex_client,
                genres=genres,
                decades=decades,
                exclude_live=exclude_live,
                min_rating=min_rating,
                max_tracks_to_ai=effective_pool_size(max_tracks_to_ai, track_count),
            )
        except PlexQueryError as e:
            yield emit("error", {"message": f"Plex server error: {e}"})
            return

        logger.info("Got %d tracks", len(filtered_tracks))

        # Seasonal filter: keywords now, AI screening during selection. Skipped
        # when the request itself is for holiday music.
        screen_seasonal = exclude_seasonal
        if exclude_seasonal and request_is_seasonal(" ".join(filter(None, [prompt, additional_notes]))):
            screen_seasonal = False
            yield emit("progress", {
                "step": "filtering",
                "message": "Your request mentions holidays, so seasonal songs are allowed.",
            })
        if screen_seasonal:
            before = len(filtered_tracks)
            filtered_tracks = [t for t in filtered_tracks if not is_seasonal(t)]
            skipped = before - len(filtered_tracks)
            logger.info("Excluded %d seasonal tracks by keyword", skipped)
            if skipped:
                yield emit("progress", {
                    "step": "filtering",
                    "message": f"Skipped {skipped} holiday/seasonal tracks...",
                })

        if not filtered_tracks:
            yield emit("error", {"message": "No tracks match the selected filters. Try broadening your selection."})
            return

        if seed_track:
            filtered_tracks = [t for t in filtered_tracks if t.rating_key != seed_track.rating_key]

        # Step 2: Report track count (sampling already done server-side)
        if has_filters:
            yield emit("progress", {"step": "filtering", "message": f"Using {len(filtered_tracks)} tracks..."})
        else:
            yield emit("progress", {"step": "filtering", "message": f"Using {len(filtered_tracks)} random tracks..."})

        # Step 3: Build the request description shared by every batch call
        request_text = _build_request_text(
            prompt, seed_track, selected_dimensions, additional_notes, refinement_answers
        )

        # Step 4: Batched selection (runs in a worker thread; progress is
        # relayed through a queue so it can be streamed while the LLM works)
        yield emit("progress", {"step": "ai_working", "message": "AI is curating your playlist..."})
        selector = BatchedSelector(llm_client, request_text, exclude_seasonal=screen_seasonal)
        selection: SelectionResult | None = None
        for kind, payload in _run_selection(selector, filtered_tracks, track_count):
            if kind == "event":
                yield emit("progress", payload)
            elif kind == "heartbeat":
                yield ": heartbeat\n\n"
            elif kind == "error":
                raise payload
            else:
                selection = payload

        assert selection is not None
        logger.info(
            "Selection complete: %d tracks, %d rounds, %d LLM calls, %d tokens, "
            "%d fallback batches, %d seasonal tracks replaced",
            len(selection.tracks), selection.rounds, selection.llm_calls,
            selection.total_tokens, selection.fallback_batches, selection.seasonal_removed,
        )
        if selection.seasonal_removed:
            yield emit("progress", {
                "step": "ai_working",
                "message": f"Replaced {selection.seasonal_removed} holiday songs the AI spotted...",
            })
        if selection.llm_calls and selection.fallback_batches == selection.llm_calls:
            yield emit("error", {
                "message": "The AI did not return any usable selections. Check that the "
                           "model is loaded and responding, or try a different model."
            })
            return

        matched_tracks: list[Track] = selection.tracks
        track_reasons: dict[str, str] = selection.reasons
        response = LLMResponse(
            content="",
            input_tokens=selection.input_tokens,
            output_tokens=selection.output_tokens,
            model=llm_client.config.model_generation,
        )
        track_selections = [
            {
                "artist": t.artist,
                "title": t.title,
                "album": t.album,
                "reason": track_reasons.get(t.rating_key, ""),
            }
            for t in matched_tracks
        ]

        # Step 7: Generate narrative
        yield emit("progress", {"step": "narrative", "message": "Writing playlist narrative..."})

        playlist_title, narrative = generate_narrative(track_selections, llm_client, prompt or "")
        logger.info("Generated narrative: title='%s', narrative_len=%d", playlist_title, len(narrative))

        # Emit narrative event for frontend
        yield emit("narrative", {
            "playlist_title": playlist_title,
            "narrative": narrative,
            "track_reasons": track_reasons,
            "user_request": prompt or "",
        })

        # Step 8: Complete
        logger.info("Track matching complete. Matched %d tracks", len(matched_tracks))
        logger.info("Emitting 'Playlist ready!' progress event")
        yield emit("progress", {"step": "complete", "message": "Playlist ready!"})

        logger.info("Building GenerateResponse: tokens=%s, cost=%s",
                    getattr(response, 'total_tokens', 'N/A'),
                    response.estimated_cost() if response else 'N/A')

        try:
            result = GenerateResponse(
                tracks=matched_tracks,
                token_count=response.total_tokens,
                estimated_cost=response.estimated_cost(),
                playlist_title=playlist_title,
                narrative=narrative,
                track_reasons=track_reasons,
            )
            logger.info("GenerateResponse built successfully with %d tracks", len(result.tracks))
        except Exception as e:
            logger.exception("Failed to build GenerateResponse: %s", e)
            yield emit("error", {"message": f"Failed to build response: {e}"})
            return

        # Send tracks in batches to work around iOS Safari dropping large SSE events.
        # Safari Mobile has undocumented buffering limits that can cause large events
        # to be silently dropped. Batching keeps each event small (~2KB).
        tracks_data = [t.model_dump(mode="json") for t in result.tracks]
        batch_size = 5
        for i in range(0, len(tracks_data), batch_size):
            batch = tracks_data[i:i + batch_size]
            logger.info("Emitting track batch %d-%d", i, i + len(batch))
            yield emit("tracks", {"batch": batch, "index": i})

        # Save result to history before emitting the final event
        result_id = None
        try:
            result_type = "seed_playlist" if seed_track else "prompt_playlist"
            # Title: always use the LLM-generated playlist title
            result_title = playlist_title
            # Use first track's rating key for card thumbnail
            first_art_key = matched_tracks[0].rating_key if matched_tracks else None
            # Subtitle: seed playlists show origin track, prompt playlists show prompt + count
            if seed_track:
                result_subtitle = f"From: {seed_track.title} by {seed_track.artist} \u00b7 {len(matched_tracks)} tracks"
            elif prompt:
                result_subtitle = f"{prompt} \u00b7 {len(matched_tracks)} tracks"
            else:
                result_subtitle = f"{len(matched_tracks)} tracks"
            result_id = library_cache.save_result(
                result_type=result_type,
                title=result_title,
                prompt=prompt or "",
                snapshot=result.model_dump(mode="json"),
                track_count=len(matched_tracks),
                art_rating_key=first_art_key,
                subtitle=result_subtitle,
            )
        except Exception as e:
            logger.warning("Failed to save result: %s", e)

        # Emit small complete event with just metadata
        logger.info("Emitting complete event")
        complete_data = {
            "track_count": len(result.tracks),
            "token_count": result.token_count,
            "estimated_cost": result.estimated_cost,
            "playlist_title": result.playlist_title,
            "narrative": result.narrative,
            "track_reasons": result.track_reasons,
        }
        if result_id:
            complete_data["result_id"] = result_id
        yield emit("complete", complete_data)
        logger.info("Complete event emitted successfully")

        # Trailing padding to push complete event through network buffers (iOS Safari fix)
        # SSE comments (lines starting with ':') are ignored by the parser but help flush buffers
        yield ": heartbeat\n\n"

    except Exception as e:
        logger.exception("Error during playlist generation")
        yield emit("error", {"message": str(e)})


NARRATIVE_SYSTEM = """You are a music connoisseur writing a brief liner note for a playlist.

Given the user's original request and the track selections (with reasons), create:
1. A creative playlist title (2-5 words, evocative, do NOT include any date)
2. A brief narrative (3 sentences, under 400 characters) that:
   - Reflects the mood or theme the user asked for
   - Mentions 3-4 specific songs by name (use single quotes around song names, e.g. 'Skinny Love')

Sound like a passionate music lover. Be concise.

Return ONLY valid JSON:
{"title": "Creative Title Here", "narrative": "Your brief narrative with 'song names' in single quotes..."}

No markdown formatting, no explanations - just the JSON object."""


def _build_request_text(
    prompt: str | None,
    seed_track: Track | None,
    selected_dimensions: list[str] | None,
    additional_notes: str | None,
    refinement_answers: list[str | None] | None,
) -> str:
    """Describe what the user wants; repeated at the top of every batch call."""
    parts = []
    if prompt:
        parts.append(f"User's request: {prompt}")
    if seed_track:
        parts.append(
            f"Seed track: {seed_track.title} by {seed_track.artist} "
            f"(from {seed_track.album}, {seed_track.year or 'Unknown year'})"
        )
        if selected_dimensions:
            parts.append(f"Explore these dimensions: {', '.join(selected_dimensions)}")
    if additional_notes:
        parts.append(f"Additional notes: {additional_notes}")
    if refinement_answers:
        answered = [a for a in refinement_answers if a]
        if answered:
            parts.append(f"User preferences: {', '.join(answered)}")
    return "\n".join(parts)


HEARTBEAT_SECONDS = 15


def _run_selection(
    selector: BatchedSelector, pool: list[Track], track_count: int
) -> Generator[tuple[str, object], None, None]:
    """Run selector.select in a thread, yielding ("event"|"heartbeat"|"error"|"done", payload)."""
    events: queue.Queue = queue.Queue()
    state = {"round": 0, "pool": 0, "keep": 0}

    def on_round(round_no: int, pool_size: int, keep: int) -> None:
        state.update(round=round_no, pool=pool_size, keep=keep)
        if keep == track_count and pool_size <= track_count * 5:
            msg = f"Picking the final {keep} from {pool_size} candidates..."
        else:
            msg = f"Round {round_no}: narrowing {pool_size:,} tracks to {keep:,}..."
        events.put(("event", {"step": "ai_working", "message": msg}))

    def on_progress(done: int, total: int) -> None:
        if total > 1:
            events.put(("event", {
                "step": "ai_working",
                "message": f"Round {state['round']}: batch {done}/{total} "
                           f"({state['pool']:,} \u2192 {state['keep']:,} tracks)",
                "batch_done": done,
                "batch_total": total,
            }))

    def on_status(message: str) -> None:
        events.put(("event", {"step": "ai_working", "message": message}))

    def worker() -> None:
        try:
            events.put(("done", selector.select(pool, track_count, on_round, on_progress, on_status)))
        except Exception as e:  # surfaced to the caller
            events.put(("error", e))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            kind, payload = events.get(timeout=HEARTBEAT_SECONDS)
        except queue.Empty:
            yield "heartbeat", None
            continue
        yield kind, payload
        if kind in ("done", "error"):
            return


def _tracks_match(llm_artist: str, llm_title: str, library_track: Track) -> bool:
    """Check if LLM selection matches a library track.

    Uses fuzzy matching to handle slight variations in naming.
    """
    from rapidfuzz import fuzz
    from backend.plex_client import simplify_string, normalize_artist, FUZZ_THRESHOLD

    # Compare titles
    simplified_llm_title = simplify_string(llm_title)
    simplified_lib_title = simplify_string(library_track.title)

    if fuzz.ratio(simplified_llm_title, simplified_lib_title) < FUZZ_THRESHOLD:
        return False

    # Compare artists (with variations)
    for artist_variant in normalize_artist(llm_artist):
        simplified_artist = simplify_string(artist_variant)
        simplified_lib_artist = simplify_string(library_track.artist)
        if fuzz.ratio(simplified_artist, simplified_lib_artist) >= FUZZ_THRESHOLD:
            return True

    return False
