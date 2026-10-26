"""Tests for Plex sign-in, sessions, admin-only settings and per-user history."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import auth, library_cache
from backend.main import app


@pytest.fixture
def sessions_db(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SESSIONS_DB", tmp_path / "sessions.db")
    monkeypatch.setattr(auth, "DATA_DIR", tmp_path)
    auth._cache.clear()
    auth._clients.clear()
    auth._pending_logins.clear()
    yield
    auth._cache.clear()


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


def _plex_tv(user, resources):
    """Fake plex.tv GET: /user then /resources."""
    def get(url, **kwargs):
        return _resp(200, user if url.endswith("/user") else resources)
    return get


AMY = {"uuid": "amy", "title": "Amy", "thumb": "https://plex.tv/amy.png", "home": False}
OWNER = {"uuid": "own", "title": "Owner", "home": True}
SERVER = {"clientIdentifier": "srv", "accessToken": "srv-token-for-user", "owned": False}


# ---------------------------------------------------------------- sign-in


class TestSignIn:
    def test_friend_with_access_signs_in_as_regular_user(self, sessions_db):
        with patch("backend.auth.httpx.get", side_effect=_plex_tv(AMY, [SERVER])):
            raw, s = auth.sign_in("amy-acct", "srv")
        assert s.user_title == "Amy" and not s.is_admin
        assert s.server_token == "srv-token-for-user"
        assert auth.get_session(raw).user_uuid == "amy"

    def test_owner_is_admin(self, sessions_db):
        owned = {**SERVER, "owned": True}
        with patch("backend.auth.httpx.get", side_effect=_plex_tv(OWNER, [owned])):
            _, s = auth.sign_in("own-acct", "srv")
        assert s.is_admin and s.owns_server and s.home

    def test_account_without_server_access_is_refused(self, sessions_db):
        other = {**SERVER, "clientIdentifier": "someone-elses"}
        with patch("backend.auth.httpx.get", side_effect=_plex_tv(AMY, [other])), \
             pytest.raises(auth.AuthError, match="doesn't have access"):
            auth.sign_in("amy-acct", "srv")

    def test_sessions_survive_restart_and_sign_out(self, sessions_db):
        with patch("backend.auth.httpx.get", side_effect=_plex_tv(AMY, [SERVER])):
            raw, _ = auth.sign_in("amy-acct", "srv")
        auth._cache.clear()  # as after a restart
        assert auth.get_session(raw).user_title == "Amy"
        auth.sign_out(raw)
        assert auth.get_session(raw) is None
        assert auth.get_session("made-up") is None

    def test_cookie_is_not_stored_in_plain_text(self, sessions_db):
        with patch("backend.auth.httpx.get", side_effect=_plex_tv(AMY, [SERVER])):
            raw, s = auth.sign_in("amy-acct", "srv")
        assert raw != s.id_hash and raw not in auth.SESSIONS_DB.read_bytes().decode("latin-1")

    def test_profile_switch_only_admin_if_account_owns_server(self, sessions_db):
        profiles = [
            {"uuid": "own", "title": "Owner", "admin": True, "protected": True, "restricted": False},
            {"uuid": "kid", "title": "Lia", "admin": False, "protected": False, "restricted": True},
        ]
        owned = {**SERVER, "owned": True}
        with patch("backend.auth.httpx.get", side_effect=_plex_tv(OWNER, [owned])):
            _, s = auth.sign_in("own-acct", "srv")
        with patch("backend.auth.plex_home.list_home_users", return_value=profiles), \
             patch("backend.auth.plex_home.switch_home_user", return_value="lia-token"):
            lia = auth.switch_profile(s, "kid", "srv", None)
        assert lia.user_title == "Lia" and not lia.is_admin and lia.server_token == "lia-token"
        with patch("backend.auth.plex_home.list_home_users", return_value=profiles), \
             patch("backend.auth.plex_home.switch_home_user", return_value="own-token"):
            back = auth.switch_profile(lia, "own", "srv", "1234")
        assert back.is_admin


# ---------------------------------------------------------------- API guard


@pytest.fixture
def signed_in(sessions_db):
    """Returns a function making a TestClient signed in as a regular user or the admin."""
    def make(admin: bool) -> TestClient:
        res = {**SERVER, "owned": admin}
        with patch("backend.auth.httpx.get", side_effect=_plex_tv(OWNER if admin else AMY, [res])):
            raw, _ = auth.sign_in("acct", "srv")
        client = TestClient(app)
        client.cookies.set(auth.SESSION_COOKIE, raw)
        return client
    return make


@pytest.fixture
def auth_on():
    with patch("backend.main.auth_enabled", return_value=True):
        yield


class TestApiGuard:
    def test_signed_out_gets_401(self, auth_on, sessions_db):
        r = TestClient(app).get("/api/results")
        assert r.status_code == 401
        assert "Sign in" in r.json()["detail"]

    def test_auth_status_and_health_are_public(self, auth_on, sessions_db):
        client = TestClient(app)
        assert client.get("/api/auth/me").json() == {"auth_required": True, "signed_in": False, "user": None}
        assert client.get("/api/health").status_code != 401

    def test_regular_user_cannot_change_settings(self, auth_on, signed_in):
        client = signed_in(admin=False)
        assert client.post("/api/config", json={"music_library": "X"}).status_code == 403
        assert client.post("/api/library/sync").status_code == 403
        assert client.post("/api/setup/complete").status_code == 403

    def test_regular_user_can_use_the_app(self, auth_on, signed_in):
        with patch("backend.main.library_cache.list_results", return_value=([], 0)) as lr:
            r = signed_in(admin=False).get("/api/results")
        assert r.status_code == 200
        assert lr.call_args.kwargs["owner"] == "amy"
        assert lr.call_args.kwargs["include_unowned"] is False

    def test_admin_reaches_settings(self, auth_on, signed_in):
        from backend.config import ConfigSaveError

        client = signed_in(admin=True)
        # Reaching the settings code (which fails to save here) proves the guard let the admin in.
        with patch("backend.main.update_config_values", side_effect=ConfigSaveError("disk full")):
            r = client.post("/api/config", json={"model_analysis": "m"})
        assert r.status_code == 500 and r.json()["detail"] == "disk full"

    def test_me_reports_user(self, auth_on, signed_in):
        me = signed_in(admin=False).get("/api/auth/me").json()
        assert me["signed_in"] and me["user"]["title"] == "Amy" and not me["user"]["is_admin"]

    def test_logout(self, auth_on, signed_in):
        client = signed_in(admin=False)
        assert client.post("/api/auth/logout").json()["signed_in"] is False
        assert client.get("/api/results").status_code == 401

    def test_sign_in_off_lets_everyone_through(self, sessions_db):
        # conftest sets DISABLE_AUTH=true
        with patch("backend.main.library_cache.list_results", return_value=([], 0)) as lr:
            assert TestClient(app).get("/api/results").status_code == 200
        assert "owner" not in lr.call_args.kwargs


class TestSignInFlow:
    def test_start_then_check_signs_the_browser_in(self, auth_on, sessions_db):
        plex = MagicMock()
        plex.is_connected.return_value = True
        plex.get_machine_identifier.return_value = "srv"
        client = TestClient(app)
        with patch("backend.auth.httpx.post", return_value=_resp(201, {"id": 7, "code": "abcd"})):
            r = client.post("/api/auth/plex/start", json={"forward_url": "http://bside/done"})
        assert r.status_code == 200
        assert "code=abcd" in r.json()["auth_url"] and "forwardUrl=" in r.json()["auth_url"]

        with patch("backend.main.get_plex_client", return_value=plex), \
             patch("backend.auth.httpx.get", return_value=_resp(200, {"authToken": None})):
            assert client.post("/api/auth/plex/check").json()["signed_in"] is False

        def plex_tv(url, **kwargs):
            if "/pins/" in url:
                return _resp(200, {"authToken": "amy-acct"})
            return _resp(200, AMY if url.endswith("/user") else [SERVER])

        with patch("backend.main.get_plex_client", return_value=plex), \
             patch("backend.auth.httpx.get", side_effect=plex_tv):
            r = client.post("/api/auth/plex/check")
        assert r.json()["signed_in"] and r.json()["user"]["title"] == "Amy"
        assert client.get("/api/auth/me").json()["signed_in"]

    def test_check_without_starting_is_refused(self, auth_on, sessions_db):
        plex = MagicMock()
        plex.is_connected.return_value = True
        with patch("backend.main.get_plex_client", return_value=plex):
            r = TestClient(app).post("/api/auth/plex/check")
        assert r.status_code == 403


# ---------------------------------------------------------------- history


class TestPerUserHistory:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
        monkeypatch.setattr(library_cache, "DB_PATH", tmp_path / "library_cache.db")

    def _save(self, owner):
        return library_cache.save_result(
            result_type="prompt_playlist", title="t", prompt="p", snapshot={}, track_count=1, owner=owner
        )

    def test_users_only_see_their_own_results(self, db):
        legacy, amys, lias = self._save(None), self._save("amy"), self._save("lia")
        ids = lambda **kw: {r["id"] for r in library_cache.list_results(**kw)[0]}
        assert ids(owner="amy") == {amys}
        assert ids(owner="own", include_unowned=True) == {legacy}
        assert ids() == {legacy, amys, lias}
        assert library_cache.get_result(lias, owner="amy") is None
        assert library_cache.delete_result(lias, owner="amy") is False
        assert library_cache.delete_result(amys, owner="amy") is True
