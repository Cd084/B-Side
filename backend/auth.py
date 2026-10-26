"""Plex sign-in and per-user sessions.

Everyone signs in with their own Plex account (the same PIN-based flow as
Overseerr/Tautulli: B-Side never sees passwords). Only accounts with access to
this Plex server get in; the server owner is the admin. Each session acts
against Plex with that user's own server access token, so their playlists are
theirs. Plex Home members can then pick a profile (incl. managed users with
no Plex login of their own), like the Plex apps' "Who's watching?".

Sessions live in data/sessions.db (survive restarts); the browser holds a
random id in an HttpOnly cookie, stored here only as a SHA-256 hash.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass, replace
from urllib.parse import quote

import httpx
import plexapi

from backend import plex_home
from backend.library_cache import DATA_DIR

logger = logging.getLogger(__name__)

PRODUCT = "B-Side"
SESSION_COOKIE = "bside_session"
LOGIN_COOKIE = "bside_login"
SESSION_MAX_IDLE = 30 * 24 * 3600  # sign in again after 30 days unused
LOGIN_PIN_TTL = 15 * 60
PLEX_TV = "https://plex.tv/api/v2"
TIMEOUT = 15.0
SESSIONS_DB = DATA_DIR / "sessions.db"


class AuthError(Exception):
    """A user-facing sign-in problem."""


@dataclass(frozen=True)
class Session:
    id_hash: str
    user_uuid: str
    user_title: str
    user_thumb: str | None
    account_token: str  # plex.tv token of who signed in (lists/switches profiles)
    server_token: str  # this user's access token for our Plex server
    is_admin: bool  # acting as the server owner
    owns_server: bool  # the signed-in account owns the server
    home: bool  # the account is in a Plex Home, so profiles can be picked


# The signed-in user for the current request (set by the auth middleware).
current_session: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    "current_session", default=None
)


def current_owner() -> str | None:
    """Plex user uuid that owns data created in this request (None = sign-in off)."""
    s = current_session.get()
    return s.user_uuid if s else None


# ---------------------------------------------------------------- plex.tv


def _headers(token: str | None = None) -> dict[str, str]:
    h = {**plexapi.BASE_HEADERS, "X-Plex-Product": PRODUCT, "X-Plex-Device-Name": PRODUCT,
         "Accept": "application/json"}
    if token:
        h["X-Plex-Token"] = token
    return h


def _client_id() -> str:
    return plexapi.BASE_HEADERS["X-Plex-Client-Identifier"]


def create_login(forward_url: str | None) -> tuple[str, str]:
    """Starts a Plex sign-in. Returns (login_nonce, auth_url).

    The browser opens auth_url (Plex's own sign-in page); B-Side then polls
    the PIN until Plex hands over a token. The nonce ties the pending sign-in
    to the browser that started it (kept in a short-lived cookie).
    """
    try:
        r = httpx.post(f"{PLEX_TV}/pins", params={"strong": "true"}, headers=_headers(), timeout=TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise AuthError(f"Could not reach plex.tv: {e}") from e
    pin = r.json()
    nonce = secrets.token_urlsafe(24)
    with _lock:
        _purge_logins()
        _pending_logins[nonce] = (pin["id"], time.time())
    url = (
        "https://app.plex.tv/auth#?"
        f"clientID={quote(_client_id())}&code={quote(pin['code'])}"
        f"&context%5Bdevice%5D%5Bproduct%5D={quote(PRODUCT)}"
    )
    if forward_url:
        url += f"&forwardUrl={quote(forward_url, safe='')}"
    return nonce, url


def poll_login(nonce: str | None) -> str | None:
    """The Plex account token once the user has signed in, else None."""
    with _lock:
        _purge_logins()
        entry = _pending_logins.get(nonce or "")
    if not entry:
        raise AuthError("Sign-in expired. Please try again.")
    try:
        r = httpx.get(f"{PLEX_TV}/pins/{entry[0]}", headers=_headers(), timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AuthError(f"Could not reach plex.tv: {e}") from e
    token = r.json().get("authToken") if r.status_code == 200 else None
    if token:
        with _lock:
            _pending_logins.pop(nonce, None)
    return token or None


def sign_in(account_token: str, server_id: str) -> tuple[str, Session]:
    """Creates a session for a Plex account. Returns (cookie value, session).

    Refuses accounts without access to this server.
    """
    try:
        me = httpx.get(f"{PLEX_TV}/user", headers=_headers(account_token), timeout=TIMEOUT)
        res = httpx.get(f"{PLEX_TV}/resources", params={"includeHttps": 1},
                        headers=_headers(account_token), timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AuthError(f"Could not reach plex.tv: {e}") from e
    if me.status_code != 200 or res.status_code != 200:
        raise AuthError("Plex did not accept the sign-in. Please try again.")
    user = me.json()
    server = next((r for r in res.json() if r.get("clientIdentifier") == server_id), None)
    if not server or not server.get("accessToken"):
        raise AuthError(
            f"{user.get('title') or 'This Plex account'} doesn't have access to this Plex server. "
            "Ask the server owner to share their music library with you."
        )
    owned = bool(server.get("owned"))
    raw = secrets.token_urlsafe(32)
    session = Session(
        id_hash=_hash(raw),
        user_uuid=user["uuid"],
        user_title=user.get("title") or user.get("username") or "Plex user",
        user_thumb=user.get("thumb"),
        account_token=account_token,
        server_token=server["accessToken"],
        is_admin=owned,
        owns_server=owned,
        home=bool(user.get("home")),
    )
    _save(session, new=True)
    logger.info("Signed in: %s%s", session.user_title, " (admin)" if owned else "")
    return raw, session


def list_profiles(session: Session) -> list[dict]:
    """Plex Home profiles the signed-in account can switch to."""
    if not session.home:
        return []
    return plex_home.list_home_users(session.account_token)


def switch_profile(session: Session, uuid: str, server_id: str, pin: str | None) -> Session:
    """Acts as another Plex Home profile from now on (PIN if it has one)."""
    profiles = list_profiles(session)
    profile = next((p for p in profiles if p["uuid"] == uuid), None)
    if profile is None:
        raise AuthError("No such Plex Home profile")
    try:
        token = plex_home.switch_home_user(session.account_token, uuid, server_id, pin)
    except plex_home.PlexHomeError as e:
        raise AuthError(str(e)) from e
    updated = replace(
        session,
        user_uuid=profile["uuid"],
        user_title=profile["title"],
        user_thumb=profile.get("thumb"),
        server_token=token,
        # The Home admin is the server owner only if this account owns the server.
        is_admin=session.owns_server and profile["admin"],
    )
    _save(updated)
    _clients.pop(session.id_hash, None)
    logger.info("Profile switch: %s -> %s", session.user_title, updated.user_title)
    return updated


# ---------------------------------------------------------------- sessions

_lock = threading.Lock()
_pending_logins: dict[str, tuple[int, float]] = {}
_cache: dict[str, tuple[Session, float]] = {}  # id_hash -> (session, last_seen)


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _purge_logins() -> None:
    cutoff = time.time() - LOGIN_PIN_TTL
    for k in [k for k, (_, t) in _pending_logins.items() if t < cutoff]:
        del _pending_logins[k]


def _db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SESSIONS_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            id_hash TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            user_title TEXT NOT NULL,
            user_thumb TEXT,
            account_token TEXT NOT NULL,
            server_token TEXT NOT NULL,
            is_admin INTEGER NOT NULL,
            owns_server INTEGER NOT NULL,
            home INTEGER NOT NULL,
            last_seen REAL NOT NULL
        )"""
    )
    return conn


def _save(s: Session, new: bool = False) -> None:
    now = time.time()
    with _lock, closing(_db()) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (s.id_hash, s.user_uuid, s.user_title, s.user_thumb, s.account_token, s.server_token,
             int(s.is_admin), int(s.owns_server), int(s.home), now),
        )
        _cache[s.id_hash] = (s, now)


def get_session(raw: str | None) -> Session | None:
    """The session for a cookie value, or None if missing/expired."""
    if not raw:
        return None
    h = _hash(raw)
    now = time.time()
    with _lock:
        cached = _cache.get(h)
    if cached and now - cached[1] < 3600:
        return cached[0]
    with _lock, closing(_db()) as conn, conn:
        row = conn.execute("SELECT * FROM sessions WHERE id_hash = ?", (h,)).fetchone()
        if not row or now - row["last_seen"] > SESSION_MAX_IDLE:
            if row:
                conn.execute("DELETE FROM sessions WHERE id_hash = ?", (h,))
            _cache.pop(h, None)
            return None
        conn.execute("UPDATE sessions SET last_seen = ? WHERE id_hash = ?", (now, h))
        s = Session(
            id_hash=h, user_uuid=row["user_uuid"], user_title=row["user_title"],
            user_thumb=row["user_thumb"], account_token=row["account_token"],
            server_token=row["server_token"], is_admin=bool(row["is_admin"]),
            owns_server=bool(row["owns_server"]), home=bool(row["home"]),
        )
        _cache[h] = (s, now)
        return s


def sign_out(raw: str | None) -> None:
    if not raw:
        return
    h = _hash(raw)
    with _lock, closing(_db()) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE id_hash = ?", (h,))
        _cache.pop(h, None)
    _clients.pop(h, None)


# ---------------------------------------------------------------- per-user Plex

_clients: OrderedDict[str, object] = OrderedDict()  # id_hash -> PlexClient
_MAX_CLIENTS = 32


def plex_client_for(session: Session, url: str, music_library: str):
    """A PlexClient acting as this session's user (cached per session)."""
    from backend.plex_client import PlexClient  # avoid import cycle at load time

    with _lock:
        client = _clients.get(session.id_hash)
        if client is not None and client.token == session.server_token:
            _clients.move_to_end(session.id_hash)
            return client
    client = PlexClient(url, session.server_token, music_library)
    with _lock:
        _clients[session.id_hash] = client
        while len(_clients) > _MAX_CLIENTS:
            _clients.popitem(last=False)
    return client
