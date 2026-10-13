from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.x ORM models."""

    pass


# Import all models so Alembic autogenerate metadata discovery works
from app.models.organization import Organization  # noqa: F401
from app.models.user import User  # noqa: F401
