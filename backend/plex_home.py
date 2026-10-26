"""Plex Home users: list them and switch to one, like the Plex apps' Switch User.

Needs the account (plex.tv) token. Switching returns that user's access token
for *this* server. The user's own plex.tv token is not accepted by the server,
so it is exchanged via plex.tv resources for the server access token.
"""

from __future__ import annotations

import httpx
import plexapi

PLEX_TV = "https://plex.tv/api/v2"
TIMEOUT = 15.0


class PlexHomeError(Exception):
    """A user-facing problem listing or switching Plex Home users."""


def _headers(token: str) -> dict[str, str]:
    return {**plexapi.BASE_HEADERS, "X-Plex-Token": token, "Accept": "application/json"}


def list_home_users(account_token: str) -> list[dict]:
    """Home users on the account: uuid, title, thumb, admin, protected, restricted."""
    try:
        r = httpx.get(f"{PLEX_TV}/home/users", headers=_headers(account_token), timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise PlexHomeError(f"Could not reach plex.tv: {e}") from e
    if r.status_code != 200:
        raise PlexHomeError(
            f"Could not list Plex Home users (plex.tv returned {r.status_code}). "
            "The Plex token must be your Plex account token."
        )
    return [
        {
            "uuid": u["uuid"],
            "title": u.get("title") or u.get("username") or "User",
            "thumb": u.get("thumb"),
            "admin": bool(u.get("admin")),
            "protected": bool(u.get("protected")),
            "restricted": bool(u.get("restricted")),
        }
        for u in r.json().get("users", [])
    ]


def switch_home_user(account_token: str, uuid: str, server_id: str, pin: str | None = None) -> str:
    """Switches to a Home user and returns their access token for server_id
    (the Plex server's machineIdentifier)."""
    try:
        sw = httpx.post(
            f"{PLEX_TV}/home/users/{uuid}/switch",
            params={"pin": pin} if pin else None,
            headers=_headers(account_token),
            timeout=TIMEOUT,
        )
        if sw.status_code in (401, 403):
            raise PlexHomeError("Wrong PIN" if pin else "This user needs a PIN")
        if sw.status_code >= 300:
            raise PlexHomeError(f"Could not switch user (plex.tv returned {sw.status_code})")
        user_token = sw.json().get("authToken")
        if not user_token:
            raise PlexHomeError("plex.tv did not return a token for this user")

        res = httpx.get(
            f"{PLEX_TV}/resources", params={"includeHttps": 1},
            headers=_headers(user_token), timeout=TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise PlexHomeError(f"Could not reach plex.tv: {e}") from e
    if res.status_code != 200:
        raise PlexHomeError(f"Could not look up servers for this user ({res.status_code})")
    for r in res.json():
        if r.get("clientIdentifier") == server_id and r.get("accessToken"):
            return r["accessToken"]
    raise PlexHomeError("This user does not have access to this Plex server")
