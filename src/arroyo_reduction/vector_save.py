import logging
import json
import time
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
                    dimred_model TEXT,
                    timestamp REAL,
                    total_processing_time REAL,
                    autoencoder_time REAL,
                    dimred_time REAL
                )
            ''')
            await self.db.commit()
            self._db_initialized = True

    async def save_vector(
            self,
            tiled_url: str,
            feature_vector: list[float],
            autoencoder_model: str,
            dimred_model: str,
            timestamp: float = None,
            total_processing_time: float = None,
            autoencoder_time: float = None,
            dimred_time: float = None):
        await self._init_db()
        # Convert numpy array to JSON string for storage
        vector_str = json.dumps(feature_vector)

        await self.db.execute(
            "INSERT INTO vectors (tiled_url, feature_vector, autoencoder_model, dimred_model, timestamp, total_processing_time, autoencoder_time, dimred_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (tiled_url, vector_str, autoencoder_model, dimred_model, timestamp, total_processing_time, autoencoder_time, dimred_time)
        )
        await self.db.commit()

    async def publish(self, message: LatentSpaceEvent) -> None:
        if not isinstance(message, LatentSpaceEvent):
            return None

        tiled_url = message.tiled_url
        feature_vector = message.feature_vector
        autoencoder_model = message.autoencoder_model
        dimred_model = message.dimred_model
        timestamp = message.timestamp if hasattr(message, "timestamp") else time.time()
        total_processing_time = message.total_processing_time if hasattr(message, "total_processing_time") else None
        autoencoder_time = message.autoencoder_time if hasattr(message, "autoencoder_time") else None
        dimred_time = message.dimred_time if hasattr(message, "dimred_time") else None
        
        await self.save_vector(
            tiled_url=tiled_url,
            feature_vector=feature_vector,
            autoencoder_model=autoencoder_model,
            dimred_model=dimred_model,
            timestamp=timestamp,
            total_processing_time=total_processing_time,
            autoencoder_time=autoencoder_time,
            dimred_time=dimred_time
        )