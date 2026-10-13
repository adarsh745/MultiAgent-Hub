import sys
import os
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Add backend root directory to Python path for tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client fixture."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as ac:
        yield ac
