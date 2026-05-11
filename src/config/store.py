import os

from src.core.postgres_store import PostgresRegulationRepository

store = PostgresRegulationRepository(connection_string=os.getenv("DATABASE_URL"))
