from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_access_token(subject: str | int, expires_minutes: int | None = None) -> str:
    expire_delta = timedelta(
        minutes=expires_minutes or settings.access_token_expire_minutes
    )
    return _encode_token(subject, expire_delta)


def create_refresh_token(subject: str | int, expires_days: int | None = None) -> str:
    expire_delta = timedelta(days=expires_days or settings.refresh_token_expire_days)
    return _encode_token(subject, expire_delta)


def _encode_token(subject: str | int, expire_delta: timedelta) -> str:
    to_encode: dict[str, Any] = {
        "sub": str(subject),
        "exp": datetime.utcnow() + expire_delta,
    }
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)