from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int


class TokenPayload(BaseModel):
    sub: str | None = None


class LoginRequest(BaseModel):
    username_or_email: str = Field(min_length=3)
    password: str = Field(min_length=8, max_length=128)

    model_config = ConfigDict(from_attributes=True)