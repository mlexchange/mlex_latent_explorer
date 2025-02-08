import asyncio

import websockets


async def client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        print("Client connected; waiting for messages...")
        async for message in websocket:
            print(f"Received: {message}")


if __name__ == "__main__":
    asyncio.run(client())
