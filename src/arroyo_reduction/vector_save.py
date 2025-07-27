
import asyncio

import aiosqlite
import os

from arroyo.listener import Listener
from arroyosas.zmq import ZMQFrameListener


class VectorSaveListener(Listener):
    def __init__(self, db_path="vector_results.db"):
        super().__init__()
        self.db_path = db_path
        self._db_initialized = False

    async def _init_db(self):
        if not self._db_initialized:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS vectors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_url TEXT NOT NULL,
                        vector TEXT NOT NULL
                    )
                ''')
                await db.commit()
            self._db_initialized = True

    async def save_vector(self, image_url, vector):
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO vectors (image_url, vector) VALUES (?, ?)",
                (image_url, str(vector))
            )
            await db.commit()

    async def handle_message(self, message):
        # Expect message to be a dict with 'vector' and 'image_url'
        vector = message.get("vector")
        image_url = message.get("image_url")
        if vector is not None and image_url is not None:
            await self.save_vector(image_url, vector)
            print(f"Saved vector for {image_url}")
        else:
            print("Invalid message: missing 'vector' or 'image_url'")

    async def start(self, zmq_settings):
        listener = ZMQFrameListener.from_settings(zmq_settings, self)
        await listener.start()

# Example usage:
# if __name__ == "__main__":
#     import dynaconf
#     settings = dynaconf.Dynaconf(settings_files=["settings.yaml"])
#     zmq_settings = settings.lse_operator.listener
#     vs = VectorSaveListener()
#     asyncio.run(vs.start(zmq_settings))
