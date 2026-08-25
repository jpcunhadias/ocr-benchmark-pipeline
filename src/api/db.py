from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import settings

engine = create_async_engine(
    settings.PG_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


# Provide a session context manager for use with `async with`
async def get_session() -> AsyncSession:
    return AsyncSessionLocal()
