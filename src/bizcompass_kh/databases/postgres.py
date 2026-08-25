from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class Postgres:
    def __init__(self, database_url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            echo=True,
        )

    def connect(self):
        pass

    def init_database(self):
        pass

    def get_db(self):
        pass
