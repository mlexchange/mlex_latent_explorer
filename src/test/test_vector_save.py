import aiosqlite
import pytest
import numpy as np

from src.arroyo_reduction.vector_save import VectorSavePublisher


@pytest.mark.asyncio
async def test_vector_save_listener(tmp_path):
    db_path = tmp_path / "test_vectors.db"
    publisher = VectorSavePublisher(db_path=str(db_path))

    # Simulate a message
    message = {"image_url": "http://example.com/image1.jpg", "vector": np.array([1, 2, 3, 4])}
    await publisher.publish(message)

    # Check that the data was saved
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute("SELECT image_url, vector FROM vectors") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == message["image_url"]
            # Convert the stored string back to a numpy array for comparison
            stored_vector = np.fromstring(rows[0][1][1:-1], sep=',', dtype=int)
            assert np.array_equal(stored_vector, message["vector"])

    # Test missing fields
    bad_message = {"image_url": "http://example.com/image2.jpg"}
    await publisher.publish(bad_message)  # Should not raise

    # Should still only be one row
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute("SELECT COUNT(*) FROM vectors") as cursor:
            count = (await cursor.fetchone())[0]
            assert count == 1
