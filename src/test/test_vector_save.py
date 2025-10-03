import aiosqlite
import json
import time
import pytest


from src.arroyo_reduction.schemas import LatentSpaceEvent
from src.arroyo_reduction.vector_save import VectorSavePublisher


@pytest.mark.asyncio
async def test_vector_save_listener(tmp_path):
    db_path = tmp_path / "test_vectors.db"
    publisher = VectorSavePublisher(db_path=str(db_path))
    
    # Initialize the database but don't call start() which would create a server
    await publisher._init_db()

    # Current timestamp for testing
    current_time = time.time()
    
    # Simulate a message with timing data and versions
    message = {
        "tiled_url": "http://example.com/image1.jpg",
        "feature_vector": [1, 2],
        "index": 0,
        "autoencoder_model": "model_v1:3",  # ← CHANGED: added version
        "dimred_model": "model_v2:2",        # ← CHANGED: added version
        "timestamp": current_time,
        "total_processing_time": 0.1234,
        "autoencoder_time": 0.0789,
        "dimred_time": 0.0445
    }
    message = LatentSpaceEvent(**message)
    
    # Call the publish method directly
    await publisher.publish(message)

    # Check that the data was saved, including timing fields
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute("""
            SELECT 
                tiled_url, 
                feature_vector, 
                autoencoder_model, 
                dimred_model,
                timestamp,
                total_processing_time,
                autoencoder_time,
                dimred_time
            FROM vectors
        """) as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == message.tiled_url
            assert rows[0][1] == json.dumps(message.feature_vector)
            assert rows[0][2] == message.autoencoder_model  # Now contains "model_v1:3"
            assert rows[0][3] == message.dimred_model        # Now contains "model_v2:2"
            assert rows[0][4] == current_time
            assert rows[0][5] == 0.1234
            assert rows[0][6] == 0.0789
            assert rows[0][7] == 0.0445

    # Explicitly close the database connection
    if publisher.db is not None:
        await publisher.db.close()