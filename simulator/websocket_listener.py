import asyncio

import websockets


async def client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            print(f"Received message: {message}")


# Run the client until it completes
asyncio.run(client())
