import time
import uuid
from typing import Any


class AgentTrace:
    def __init__(self, request: str):
        self.trace_id = str(uuid.uuid4())
        self.request = request
        self.start_time = time.perf_counter()

        self.events: list[dict[str, Any]] = []

    def log(self, event: str, **data: Any) -> None:
        self.events.append({
            "event": event,
            "timestamp": time.time(),
            "data": data,
        })

    def finish(self) -> dict[str, Any]:
        duration = time.perf_counter() - self.start_time

        return {
            "trace_id": self.trace_id,
            "request": self.request,
            "duration_seconds": round(duration, 4),
            "events": self.events,
        }

    def print_trace(self) -> None:
        trace = self.finish()

        print("\n" + "=" * 60)
        print("AGENT TRACE")
        print("=" * 60)

        print(f"Trace ID: {trace['trace_id']}")
        print(f"Request: {trace['request']}")
        print(f"Duration: {trace['duration_seconds']} seconds")

        print("\nEvents:")

        for index, event in enumerate(trace["events"], start=1):
            print(f"\n{index}. {event['event']}")

            for key, value in event["data"].items():
                print(f"   {key}: {value}")

        print("=" * 60)