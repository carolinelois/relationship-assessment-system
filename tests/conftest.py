import pytest
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from typing import Generator, AsyncGenerator
import aioredis
import aio_pika
from ..src.models.database import Base, get_db
from ..src.api.main import app
from ..src.utils.config import config

# Test database URL
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create test engine
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Test database session
TestingSessionLocal = Session(bind=engine)

async def override_get_db() -> Generator[Session, None, None]:
    try:
        db = TestingSessionLocal
        yield db
    finally:
        db.close()

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop test database tables
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
async def redis_client() -> AsyncGenerator[aioredis.Redis, None]:
    redis = await aioredis.from_url("redis://localhost", decode_responses=True)
    yield redis
    await redis.close()

@pytest.fixture
async def rabbitmq_connection() -> AsyncGenerator[aio_pika.Connection, None]:
    connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")
    yield connection
    await connection.close()

@pytest.fixture
async def rabbitmq_channel(
    rabbitmq_connection: aio_pika.Connection
) -> AsyncGenerator[aio_pika.Channel, None]:
    channel = await rabbitmq_connection.channel()
    yield channel
    await channel.close()

@pytest.fixture
def test_config():
    return {
        "app": {
            "name": "relationship-assessment-system-test",
            "version": "1.0.0"
        },
        "security": {
            "jwt_secret_key": "test_secret_key",
            "jwt_algorithm": "HS256",
            "access_token_expire_minutes": 30
        }
    }