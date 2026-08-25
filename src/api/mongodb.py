from motor.motor_asyncio import AsyncIOMotorClient

from .config import settings

client = AsyncIOMotorClient(settings.MONGO_URL)
db = client[settings.MONGO_DB]


async def get_mongo_db():
    return db
