"""Database setup and session management."""

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create SQLAlchemy engine
# Note: echo is disabled when silence_sqlalchemy is True, even in debug mode
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=settings.debug and not settings.silence_sqlalchemy,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.

    Yields:
        SQLAlchemy session that is automatically closed after use
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def migrate_database() -> None:
    """
    Run database migrations for schema changes.

    This function handles adding new columns to existing tables without
    requiring manual SQL or Alembic migrations. Each migration is idempotent
    (safe to run multiple times).
    """
    with engine.connect() as conn:
        # Check if we're using SQLite
        is_sqlite = "sqlite" in settings.database_url

        if is_sqlite:
            # Migration 1: Add skip_translation column to jobs table (v0.3.0)
            try:
                # Check if column exists by querying table info
                result = conn.execute(text("PRAGMA table_info(jobs)"))
                columns = [row[1] for row in result]

                missing_columns = set()
                if "skip_translation" not in columns:
                    missing_columns.add("skip_translation")
                if "length_scale" not in columns:
                    missing_columns.add("length_scale")
                if "noise_scale" not in columns:
                    missing_columns.add("noise_scale")
                if "noise_w_scale" not in columns:
                    missing_columns.add("noise_w_scale")
                if "target_output_path" not in columns:
                    missing_columns.add("target_output_path")
                if "cleanup_original" not in columns:
                    missing_columns.add("cleanup_original")

                if "skip_translation" in missing_columns:
                    logger.info("Running migration: Adding skip_translation column to jobs table")
                    conn.execute(
                        text(
                            "ALTER TABLE jobs ADD COLUMN skip_translation BOOLEAN DEFAULT 0 NOT NULL"
                        )
                    )
                    conn.commit()
                    logger.info("Migration completed successfully")
                else:
                    logger.debug("skip_translation column already exists, skipping migration")

                if "length_scale" in missing_columns:
                    logger.info("Running migration: Adding length_scale column to jobs table")
                    conn.execute(text("ALTER TABLE jobs ADD COLUMN length_scale REAL"))
                    conn.commit()
                    logger.info("Migration completed successfully")
                else:
                    logger.debug("length_scale column already exists, skipping migration")

                if "noise_scale" in missing_columns:
                    logger.info("Running migration: Adding noise_scale column to jobs table")
                    conn.execute(text("ALTER TABLE jobs ADD COLUMN noise_scale REAL"))
                    conn.commit()
                    logger.info("Migration completed successfully")
                else:
                    logger.debug("noise_scale column already exists, skipping migration")

                if "noise_w_scale" in missing_columns:
                    logger.info("Running migration: Adding noise_w_scale column to jobs table")
                    conn.execute(text("ALTER TABLE jobs ADD COLUMN noise_w_scale REAL"))
                    conn.commit()
                    logger.info("Migration completed successfully")
                else:
                    logger.debug("noise_w_scale column already exists, skipping migration")

                if "target_output_path" in missing_columns:
                    logger.info("Running migration: Adding target_output_path column to jobs table")
                    conn.execute(text("ALTER TABLE jobs ADD COLUMN target_output_path TEXT"))
                    conn.commit()
                    logger.info("Migration completed successfully")
                else:
                    logger.debug("target_output_path column already exists, skipping migration")

                if "cleanup_original" in missing_columns:
                    logger.info("Running migration: Adding cleanup_original column to jobs table")
                    conn.execute(
                        text(
                            "ALTER TABLE jobs ADD COLUMN cleanup_original BOOLEAN DEFAULT 0 NOT NULL"
                        )
                    )
                    conn.commit()
                    logger.info("Migration completed successfully")
                else:
                    logger.debug("cleanup_original column already exists, skipping migration")

            except Exception as e:
                logger.error(f"Migration failed: {e}")
                # Don't raise - allow app to continue if migration fails
                # (column might already exist from fresh install)

        else:
            # For PostgreSQL/MySQL, use different syntax if needed
            logger.warning(
                "Auto-migration only supports SQLite. Please run manual migrations for other databases."
            )
