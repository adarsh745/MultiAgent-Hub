from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Health check endpoint verifying API status and database connectivity."""
    db_status = "unknown"
    try:
        result = await db.execute(text("SELECT 1"))
        if result.scalar() == 1:
            db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "database": db_status,
    }
