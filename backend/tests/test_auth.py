import uuid
import pytest
from sqlalchemy import select
from app.db.database import AsyncSessionLocal
from app.models.user import User


@pytest.mark.asyncio
async def test_auth_full_flow(client):
    """Test all 10 authentication requirements in sequence."""
    suffix = uuid.uuid4().hex[:6]
    email = f"user_{suffix}@example.com"
    password = "securePassword123!"
    full_name = "Test Security User"

    # 1. Successful Registration & 9. Default MEMBER role & 10. Password hash not exposed
    reg_payload = {
        "email": email,
        "full_name": full_name,
        "password": password
    }
    res_reg = await client.post("/api/v1/auth/register", json=reg_payload)
    assert res_reg.status_code == 201, res_reg.text
    reg_data = res_reg.json()
    assert reg_data["email"] == email
    assert reg_data["role"] == "MEMBER"
    assert "password" not in reg_data
    assert "password_hash" not in reg_data

    # 2. Duplicate Email Registration
    res_dup = await client.post("/api/v1/auth/register", json=reg_payload)
    assert res_dup.status_code == 400
    assert "already registered" in res_dup.json()["detail"]

    # 4. Invalid Password Login
    res_bad_login = await client.post("/api/v1/auth/login", json={"email": email, "password": "wrongPassword!"})
    assert res_bad_login.status_code == 401
    assert "Invalid email or password" in res_bad_login.json()["detail"]

    # 3. Successful Login
    res_login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert res_login.status_code == 200
    login_data = res_login.json()
    assert "access_token" in login_data
    assert login_data["token_type"] == "bearer"
    token = login_data["access_token"]

    # 5. Accessing /me with valid JWT & 10. Password hash excluded
    res_me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    me_data = res_me.json()
    assert me_data["email"] == email
    assert me_data["role"] == "MEMBER"
    assert "password_hash" not in me_data
    assert "password" not in me_data

    # 6. Accessing /me without JWT
    res_no_jwt = await client.get("/api/v1/auth/me")
    assert res_no_jwt.status_code == 401

    # 7. Accessing /me with invalid JWT
    res_invalid_jwt = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid_token_abc"})
    assert res_invalid_jwt.status_code == 401

    # 8. Inactive user rejection
    # Deactivate user in DB directly
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == email))
        db_user = result.scalar_one()
        db_user.is_active = False
        await session.commit()

    # Attempting /me with inactive user token
    res_inactive_me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_inactive_me.status_code == 400
    assert "Inactive user account" in res_inactive_me.json()["detail"]

    # Attempting login with inactive user
    res_inactive_login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert res_inactive_login.status_code == 400
    assert "Inactive user account" in res_inactive_login.json()["detail"]
