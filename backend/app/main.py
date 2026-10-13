from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ai import router as ai_router
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local frontend development
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount Routers
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(health_router)  # Root level /health probe
app.include_router(auth_router, prefix=f"{settings.API_V1_STR}/auth")
app.include_router(ai_router, prefix=f"{settings.API_V1_STR}/ai")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint providing documentation links and status."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} Backend API",
        "docs": "/docs",
        "health": "/health",
    }
