

import asyncio
import logging
import json
import numpy as np
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


    async def _init_db(self):
        if not self._db_initialized:
            if self.db is None:
                self.db = await aiosqlite.connect(self.db_path)
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_url TEXT NOT NULL,
                    vector TEXT NOT NULL
                )
            ''')
            await self.db.commit()
            self._db_initialized = True

    async def save_vector(self, image_url: str, vector: np.ndarray):
        await self._init_db()
        # Convert numpy array to JSON string for storage
        vector_str = json.dumps(vector.tolist())

        await self.db.execute(
            "INSERT INTO vectors (image_url, vector) VALUES (?, ?)",
            (image_url, vector_str)
        )
        await self.db.commit()

    async def publish(self, message: LatentSpaceEvent) -> None:
        # Expect message to be a dict with 'vector' and 'image_url'
        vector = message.get("vector")
        image_url = message.get("image_url")
        if vector is not None and image_url is not None:
            await self.save_vector(image_url, vector)
            logger.debug(f"Saved vector for {image_url}")
        else:
            logger.debug("Invalid message: missing 'vector' or 'image_url'")

        

   