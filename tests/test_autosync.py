"""Tests for automatic library re-sync."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from backend import main


class _Stop(Exception):
    pass


def _run_loop(*, stale, hours=24, syncing=False, cached=True, connected=True):
    plex = MagicMock()
    plex.is_connected.return_value = connected
    config = MagicMock()
    config.plex.auto_sync_hours = hours
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise _Stop

    with patch.object(main.asyncio, "sleep", fake_sleep), \
         patch.object(main, "get_config", return_value=config), \
         patch.object(main, "get_plex_client", return_value=plex), \
         patch.object(main.library_cache, "has_cached_tracks", return_value=cached), \
         patch.object(main.library_cache, "get_sync_progress", return_value={"is_syncing": syncing}), \
         patch.object(main.library_cache, "is_cache_stale", return_value=stale) as is_stale, \
         patch.object(main.library_cache, "sync_library") as sync:
        with pytest.raises(_Stop):
            asyncio.run(main._auto_sync_loop())
    return sync, is_stale


def test_stale_cache_is_resynced():
    sync, is_stale = _run_loop(stale=True)
    sync.assert_called_once()
    is_stale.assert_called_once_with(max_age_hours=24)


@pytest.mark.parametrize("kwargs", [
    {"stale": False},
    {"stale": True, "hours": 0},          # disabled
    {"stale": True, "syncing": True},     # already running
    {"stale": True, "cached": False},     # first sync belongs to setup
    {"stale": True, "connected": False},
])
def test_no_resync(kwargs):
    sync, _ = _run_loop(**kwargs)
    sync.assert_not_called()
