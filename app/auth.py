"""
Minimal role-based access control (RBAC) for the demo.

IMPORTANT (see README "Security note"): the role is taken directly from
an unsigned request header. This is sufficient to demonstrate
retrieval-time access filtering for a portfolio project, but it is NOT
real authentication — anyone can claim any role. Replace with signed
JWT/session-based auth before using this with real access-controlled
documents.
"""
from fastapi import Header, HTTPException, status

from app.config import settings


def get_current_role(x_role: str = Header(default="public")) -> str:
    role = x_role.lower().strip()
    if role not in settings.role_access_levels:
        allowed = ", ".join(settings.role_access_levels.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown role '{x_role}'. Allowed roles: {allowed}",
        )
    return role


def allowed_access_levels(role: str) -> list[str]:
    return settings.role_access_levels[role]
