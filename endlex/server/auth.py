"""Bearer-token auth dependencies.

Single token for the whole server, read from ``$ENDLEX_TOKEN``. Writes always
require it. Reads are open by default; set ``ENDLEX_PUBLIC_READS=0`` to require
the same token on reads too.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def _expected_token() -> str:
    tok = os.environ.get("ENDLEX_TOKEN", "")
    if not tok:
        # Misconfigured server. Surface as 500 so the operator notices
        # rather than silently letting writes through.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "server misconfigured: ENDLEX_TOKEN unset",
        )
    return tok


def _check_bearer(authorization: str | None) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token"
        )
    presented = authorization.split(None, 1)[1].strip()
    if presented != _expected_token():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bad token")


def require_write_auth(authorization: str | None = Header(default=None)) -> None:
    _check_bearer(authorization)


def require_read_auth(authorization: str | None = Header(default=None)) -> None:
    if os.environ.get("ENDLEX_PUBLIC_READS", "1") == "1":
        return
    _check_bearer(authorization)
