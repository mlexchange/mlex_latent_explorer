import asyncio

import websockets

clients = set()


async def handler(websocket):
    clients.add(websocket)
    try:
        async for message in websocket:
            # Echo the message back or broadcast
            for ws in clients:
                await ws.send(message)
    finally:
        clients.remove(websocket)


async def main():
    print("Starting WebSocket server on ws://localhost:8765...")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
