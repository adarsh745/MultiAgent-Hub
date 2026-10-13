from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.security import create_access_token, get_password_hash, verify_password
from app.db.database import get_db
from app.models.organization import Organization
from app.models.user import User, UserRole
from app.schemas.auth import TokenResponse, UserLogin, UserRegister, UserResponse

router = APIRouter(tags=["Auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserRegister,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Register a new user account with default MEMBER role."""
    # Check if user with email already exists
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Hash password securely
    hashed_pwd = get_password_hash(payload.password)

    # Attach or create organization
    org_id = None
    if payload.organization_name:
        org = Organization(name=payload.organization_name)
        db.add(org)
        await db.commit()
        await db.refresh(org)
        org_id = org.id
    else:
        result = await db.execute(select(Organization).limit(1))
        default_org = result.scalar_one_or_none()
        if not default_org:
            default_org = Organization(name="Default Organization")
            db.add(default_org)
            await db.commit()
            await db.refresh(default_org)
        org_id = default_org.id

    # Create user with default MEMBER role
    user = User(
        organization_id=org_id,
        email=payload.email,
        full_name=payload.full_name,
        password_hash=hashed_pwd,
        role=UserRole.MEMBER,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login_user(
    payload: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate user email and password, returning JWT access token."""
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account",
        )

    # Issue JWT token
    access_token = create_access_token(
        subject=user.id,
        extra_claims={"role": user.role.value, "org_id": user.organization_id},
    )

    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Retrieve profile information of the currently authenticated user."""
    return UserResponse.model_validate(current_user)
