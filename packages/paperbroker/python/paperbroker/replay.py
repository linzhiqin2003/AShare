"""Merge recorded quote batches and order-script commands by explicit event time."""

from collections.abc import Iterable

from .api import Broker, Event, QuoteArray


def replay(
    broker: Broker,
    recordings: Iterable[tuple[int, QuoteArray]],
    script: Iterable[Event],
) -> list[Event]:
    """Replay commands before quotes at equal receipt times, preserving script order."""
    commands = iter(sorted(script, key=lambda event: event["time"]))
    pending = next(commands, None)
    events = []
    previous = -1
    for received, quotes in recordings:
        if received < previous:
            raise ValueError("recordings must be ordered by receipt time")
        previous = received
        while pending is not None and pending["time"] <= received:
            broker.apply(pending)
            events.extend(broker.events())
            pending = next(commands, None)
        broker.feed(quotes, received_at=received)
        events.extend(broker.events())
    while pending is not None:
        broker.apply(pending)
        events.extend(broker.events())
        pending = next(commands, None)
    return events
