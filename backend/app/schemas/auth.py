from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

from app.models.user import UserRole


class UserRegister(BaseModel):
    """Registration request payload."""

    email: EmailStr
    full_name: Optional[str] = Field(default=None, max_length=255)
    password: str = Field(min_length=8, max_length=128)
    organization_name: Optional[str] = Field(default=None, max_length=255)


class UserLogin(BaseModel):
    """Login request payload."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Safe user representation returned in API responses (password_hash excluded)."""

    id: int
    organization_id: Optional[int] = None
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class TokenResponse(BaseModel):
    """JWT bearer token response."""

    access_token: str
    token_type: str = "bearer"
