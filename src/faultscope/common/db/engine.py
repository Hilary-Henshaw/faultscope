"""SQLAlchemy 2.0 async engine and session factory for FaultScope.

Call ``initialize_engine`` once at service startup with the resolved
``DatabaseSettings``.  Afterwards, use ``create_async_session`` as an
async context manager or FastAPI dependency to obtain a scoped
``AsyncSession``.

Example (FastAPI)::

    from faultscope.common.db.engine import (
        initialize_engine,
        create_async_session,
    )
    from faultscope.common.config import DatabaseSettings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_engine(DatabaseSettings())
        yield

    @router.get("/health")
    async def health(db: AsyncSession = Depends(create_async_session)):
        ...
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from faultscope.common.config import DatabaseSettings
from faultscope.common.exceptions import (
    ConfigurationError,
    DatabaseError,
)
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# Module-level singletons.  These are intentionally not thread-safe at
# initialisation time; call ``initialize_engine`` before spawning any
# async tasks that use the session factory.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def initialize_engine(settings: DatabaseSettings) -> None:
    """Create and configure the async SQLAlchemy engine.

    Must be called exactly once at service startup before any call to
    ``create_async_session``.  Calling it a second time replaces the
    existing engine (the old engine is *not* disposed; callers are
    responsible for cleanup if hot-reloading is required).

    Parameters
    ----------
    settings:
        Resolved ``DatabaseSettings`` instance containing the DSN,
        pool size, and overflow configuration.
    """
    global _engine, _session_factory  # noqa: PLW0603

    dsn = settings.async_url
    _log.info(
        "db_engine_initializing",
        host=settings.host,
        port=settings.port,
        name=settings.name,
        user=settings.user,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
    )

    _engine = create_async_engine(
        dsn,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        # Return connections to pool when idle, rather than keeping
        # them open indefinitely.
        pool_recycle=3600,
        # Emit a warning when a connection is checked out for longer
        # than 30 s (helps detect connection leaks in dev).
        pool_timeout=30,
        # Echo SQL only at DEBUG level; never in production.
        echo=False,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    _log.info(
        "db_engine_initialized",
        host=settings.host,
        port=settings.port,
        name=settings.name,
    )


async def create_async_session() -> AsyncIterator[AsyncSession]:
    """Yield a database session for a single unit of work.

    Designed to be used as an async context manager or as a FastAPI
    ``Depends`` dependency.  The session is automatically closed when
    the ``async with`` block exits.  The caller is responsible for
    committing or rolling back the transaction.

    Yields
    ------
    AsyncSession
        An open ``AsyncSession`` bound to the engine created by
        ``initialize_engine``.

    Raises
    ------
    ConfigurationError
        If ``initialize_engine`` has not been called.
    DatabaseError
        If the session cannot be acquired from the pool.

    Example::

        async with create_async_session() as session:
            result = await session.execute(select(Machine))
    """
    if _session_factory is None:
        raise ConfigurationError(
            "Database engine not initialised. "
            "Call initialize_engine() at startup.",
            context={"hint": "initialize_engine must be called first"},
        )

    session: AsyncSession = _session_factory()
    try:
        yield session
    except Exception as exc:
        _log.error(
            "db_session_error",
            error=str(exc),
            exc_info=True,
        )
        await session.rollback()
        raise DatabaseError(
            f"Database session error: {exc}",
            context={"error": str(exc)},
        ) from exc
    finally:
        await session.close()


async def check_connection() -> bool:
    """Verify database connectivity by executing a trivial query.

    Used by health-check endpoints and readiness probes.  Logs the
    failure at ``ERROR`` level but does not raise; callers should
    inspect the return value.

    Returns
    -------
    bool
        ``True`` if the database is reachable and responsive,
        ``False`` otherwise.

    Raises
    ------
    ConfigurationError
        If ``initialize_engine`` has not been called.
    """
    if _engine is None:
        raise ConfigurationError(
            "Database engine not initialised. "
            "Call initialize_engine() at startup.",
            context={"hint": "initialize_engine must be called first"},
        )

    try:
        async with _engine.connect() as conn:
            from sqlalchemy import text

            await conn.execute(text("SELECT 1"))
        _log.debug("db_connection_check_passed")
        return True
    except Exception as exc:  # noqa: BLE001
        _log.error(
            "db_connection_check_failed",
            error=str(exc),
        )
        return False
