"""Minimal Endee client for local / Docker (no auth by default)."""

from __future__ import annotations

import os

from endee import Endee


def resolve_endee_client() -> tuple[Endee, str, str]:
    
    raw = (os.environ.get("ENDEE_BASE_URL") or "http://127.0.0.1:18080").strip().rstrip("/")
    if raw.endswith("/api/v1"):
        api_base = raw
    else:
        api_base = raw + "/api/v1"

    token = (os.environ.get("ENDEE_API_KEY") or os.environ.get("NDD_AUTH_TOKEN") or "").strip()
    if os.environ.get("ENDEE_API_KEY"):
        auth_mode = "ENDEE_API_KEY"
    elif os.environ.get("NDD_AUTH_TOKEN"):
        auth_mode = "NDD_AUTH_TOKEN"
    else:
        auth_mode = "none"

    client = Endee(token=token if token else "")
    client.set_base_url(api_base)
    return client, api_base, auth_mode
