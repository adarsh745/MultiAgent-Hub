"""Add password_hash and role enum to users table

Revision ID: 002_add_auth_fields
Revises: 001_initial_schema
Create Date: 2026-09-14 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '002_add_auth_fields'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create PostgreSQL Enum type
    user_role_enum = postgresql.ENUM('ADMIN', 'MANAGER', 'MEMBER', name='user_role_enum')
    user_role_enum.create(op.get_bind(), checkfirst=True)

    # Add password_hash and role columns to users
    op.add_column('users', sa.Column('password_hash', sa.String(length=255), server_default='', nullable=False))
    op.add_column('users', sa.Column('role', sa.Enum('ADMIN', 'MANAGER', 'MEMBER', name='user_role_enum'), server_default='MEMBER', nullable=False))

    # Make organization_id nullable
    op.alter_column('users', 'organization_id', existing_type=sa.BigInteger(), nullable=True)


def downgrade() -> None:
    op.alter_column('users', 'organization_id', existing_type=sa.BigInteger(), nullable=False)
    op.drop_column('users', 'role')
    op.drop_column('users', 'password_hash')

    user_role_enum = postgresql.ENUM('ADMIN', 'MANAGER', 'MEMBER', name='user_role_enum')
    user_role_enum.drop(op.get_bind(), checkfirst=True)
