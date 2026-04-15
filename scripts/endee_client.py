"""Resolve Endee client + API base for benchmark / ingest scripts."""

from __future__ import annotations

import os

from endee import Endee


def resolve_endee_client() -> tuple[Endee, str, str]:
    """
    Returns (client, api_base_url, auth_mode_label).

    Env:
      ENDEE_BASE_URL — e.g. http://host:18080 (script appends /api/v1 if missing)
      ENDEE_API_KEY or NDD_AUTH_TOKEN — optional bearer token for cloud / auth installs
      ENDEE_LOCAL_NO_AUTH — set to 1 if the server accepts an empty token (local dev)
    """
    raw = (os.environ.get("ENDEE_BASE_URL") or "").strip().rstrip("/")
    if not raw:
        raise SystemExit("Set ENDEE_BASE_URL (see .env example)")

    if raw.endswith("/api/v1"):
        api_base = raw
    else:
        api_base = raw + "/api/v1"

    token = (os.environ.get("ENDEE_API_KEY") or os.environ.get("NDD_AUTH_TOKEN") or "").strip()
    local = os.environ.get("ENDEE_LOCAL_NO_AUTH", "").lower() in ("1", "true", "yes")

    if token:
        auth_mode = "ENDEE_API_KEY" if os.environ.get("ENDEE_API_KEY") else "NDD_AUTH_TOKEN"
        client = Endee(token=token)
    elif local:
        auth_mode = "local_no_auth"
        client = Endee(token="")
    else:
        raise SystemExit(
            "Set ENDEE_API_KEY or NDD_AUTH_TOKEN, or ENDEE_LOCAL_NO_AUTH=1 for tokenless local",
        )

    client.set_base_url(api_base)
    return client, api_base, auth_mode
