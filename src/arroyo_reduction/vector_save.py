import logging
import json
import aiosqlite

from arroyopy.operator import Operator
from arroyopy.publisher import Publisher

from .schemas import LatentSpaceEvent

logger = logging.getLogger("arroyo_reduction.vector_save")

class VectorSavePublisher(Publisher):
    def __init__(self, db_path="vector_results.db"):
        super().__init__()
        self.db_path = db_path
        self._db_initialized = False
        self.db: aiosqlite.Connection = None
        # Database will be initialized lazily in start()

    async def start(self):
        logger.info(f"Starting VectorSavePublisher with DB path: {self.db_path}")
        await self._init_db()

    async def _init_db(self):
        if not self._db_initialized:
            if self.db is None:
                self.db = await aiosqlite.connect(self.db_path)
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tiled_url TEXT NOT NULL,
                    feature_vector TEXT NOT NULL,
                    autoencoder_model TEXT,
                    dimred_model TEXT
                )
            ''')
            await self.db.commit()
            self._db_initialized = True

    async def save_vector(
            self,
            tiled_url: str,
            feature_vector: list[float],
            autoencoder_model: str,
            dimred_model: str):
        await self._init_db()
        # Convert numpy array to JSON string for storage
        vector_str = json.dumps(feature_vector)

        await self.db.execute(
            "INSERT INTO vectors (tiled_url, feature_vector, autoencoder_model, dimred_model) VALUES (?, ?, ?, ?)",
            (tiled_url, vector_str, autoencoder_model, dimred_model)
        )
        await self.db.commit()

    async def publish(self, message: LatentSpaceEvent) -> None:
        if not isinstance(message, LatentSpaceEvent):
            return None

        tiled_url = message.tiled_url
        feature_vector = message.feature_vector
        autoencoder_model = message.autoencoder_model
        dimred_model = message.dimred_model
        await self.save_vector(
            tiled_url=tiled_url,
            feature_vector=feature_vector,
            autoencoder_model=autoencoder_model,
            dimred_model=dimred_model
        )
        