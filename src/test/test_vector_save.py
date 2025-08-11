import aiosqlite
import json
import pytest


from src.arroyo_reduction.schemas import LatentSpaceEvent
from src.arroyo_reduction.vector_save import VectorSavePublisher


@pytest.mark.asyncio
async def test_vector_save_listener(tmp_path):
    db_path = tmp_path / "test_vectors.db"
    publisher = VectorSavePublisher(db_path=str(db_path))
    
    # Initialize the database but don't call start() which would create a server
    await publisher._init_db()

    # Simulate a message
    message = {
        "tiled_url": "http://example.com/image1.jpg",
        "feature_vector": [1, 2],
        "index": 0,
        "autoencoder_model": "model_v1",
        "dimred_model": "model_v2"
    }
    message = LatentSpaceEvent(**message)
    
    # Call the publish method directly
    await publisher.publish(message)

    # Check that the data was saved
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute("SELECT tiled_url, feature_vector, autoencoder_model, dimred_model FROM vectors") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == message.tiled_url
            assert rows[0][1] == json.dumps(message.feature_vector)
            assert rows[0][2] == message.autoencoder_model
            assert rows[0][3] == message.dimred_model

    # Explicitly close the database connection
    if publisher.db is not None:
        await publisher.db.close()
    


    # # Should still only be one row
    # async with aiosqlite.connect(str(db_path)) as db:
    #     async with db.execute("SELECT COUNT(*) FROM vectors") as cursor:
    #         count = (await cursor.fetchone())[0]
    #         assert count == 1
