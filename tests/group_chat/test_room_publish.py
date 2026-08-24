"""Group room SSE fan-out should not drop live listeners."""

from __future__ import annotations

import asyncio

from light_house.group_chat import room


def test_publish_drops_oldest_instead_of_listener() -> None:
    async def _run() -> None:
        # Isolate module subscriber set for this test.
        room._subscribers.clear()
        q = await room.subscribe()
        # Fill to capacity.
        for i in range(q.maxsize):
            q.put_nowait({"type": "fill", "n": i})
        assert q.full()
        await room.publish_room_event({"type": "utterance", "text": "still here"})
        # Subscriber remains; newest event is present after dropping oldest.
        assert q in room._subscribers
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert any(item.get("type") == "utterance" for item in items)
        await room.unsubscribe(q)

    asyncio.run(_run())
