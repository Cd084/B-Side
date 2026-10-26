"""Tests for the plex.tv Plex Home calls (list profiles, switch to one)."""

from unittest.mock import MagicMock, patch

import pytest

from backend import plex_home


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


class TestPlexTv:
    def test_list_home_users(self):
        payload = {"users": [{"uuid": "a", "title": "Owner", "admin": True, "protected": True},
                             {"uuid": "b", "title": "Amy", "restricted": True}]}
        with patch("backend.plex_home.httpx.get", return_value=_resp(200, payload)):
            users = plex_home.list_home_users("acct")
        assert [u["title"] for u in users] == ["Owner", "Amy"]
        assert users[0]["admin"] and users[0]["protected"] and users[1]["restricted"]

    def test_list_home_users_rejects_non_account_token(self):
        with patch("backend.plex_home.httpx.get", return_value=_resp(401)), \
             pytest.raises(plex_home.PlexHomeError, match="account token"):
            plex_home.list_home_users("server-only-token")

    def test_switch_returns_server_access_token(self):
        resources = [{"clientIdentifier": "other", "accessToken": "x"},
                     {"clientIdentifier": "srv", "accessToken": "amy-server-token"}]
        with patch("backend.plex_home.httpx.post", return_value=_resp(201, {"authToken": "amy-tv"})), \
             patch("backend.plex_home.httpx.get", return_value=_resp(200, resources)) as get:
            token = plex_home.switch_home_user("acct", "b", "srv")
        assert token == "amy-server-token"
        assert get.call_args.kwargs["headers"]["X-Plex-Token"] == "amy-tv"

    def test_wrong_pin(self):
        with patch("backend.plex_home.httpx.post", return_value=_resp(403)), \
             pytest.raises(plex_home.PlexHomeError, match="Wrong PIN"):
            plex_home.switch_home_user("acct", "a", "srv", pin="1234")

    def test_user_without_server_access(self):
        with patch("backend.plex_home.httpx.post", return_value=_resp(201, {"authToken": "t"})), \
             patch("backend.plex_home.httpx.get", return_value=_resp(200, [])), \
             pytest.raises(plex_home.PlexHomeError, match="access to this Plex server"):
            plex_home.switch_home_user("acct", "b", "srv")
