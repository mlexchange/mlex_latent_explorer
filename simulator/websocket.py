import asyncio
import os

import aio_pika
import websockets
from dotenv import load_dotenv

load_dotenv(".env")

WEBSOCKET_PORT = os.getenv("WEBSOCKET_PORT", 8765)
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "localhost")

# Set of connected WebSocket clients
clients = set()


async def register(websocket):
    clients.add(websocket)


async def unregister(websocket):
    clients.remove(websocket)


async def handler(websocket, path):
    await register(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        await unregister(websocket)


async def main():
    # Set up RabbitMQ connection
    connection = await aio_pika.connect_robust(f"amqp://guest:guest@{WEBSOCKET_URL}/")
    async with connection:
        # Creating channel
        channel = await connection.channel()
        # Declaring queue
        queue = await channel.declare_queue("latent_space_explorer", auto_delete=True)

        # Start the WebSocket server
        start_server = websockets.serve(handler, WEBSOCKET_URL, WEBSOCKET_PORT)

        # Run the WebSocket server and the RabbitMQ client concurrently
        await asyncio.gather(
            start_server,
            forward_messages(queue),
        )


async def forward_messages(queue):
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                # Forward the message from RabbitMQ to all WebSocket clients
                for websocket in clients:
                    await websocket.send(message.body.decode())


# Run the main function until it completes
asyncio.run(main())
