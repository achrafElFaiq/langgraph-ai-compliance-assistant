import os

from src.infrastructure.store.postgres_store import PostgresRegulationRepository

store = PostgresRegulationRepository(connection_string=os.getenv("DATABASE_URL"))
