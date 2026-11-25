"""Small offline examples accompanying the ML system-design explanations."""

import asyncio
import hashlib
import math


def capacity_estimate(
    daily_requests, peak_multiplier, service_seconds, target_utilization=0.7
):
    if daily_requests < 0 or peak_multiplier < 1 or service_seconds < 0:
        raise ValueError("invalid traffic or service-time assumption")
    if not 0 < target_utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    peak_per_second = daily_requests / 86400 * peak_multiplier
    # Assumes one concurrent request per slot and that service_seconds is representative.
    slots = math.ceil(peak_per_second * service_seconds / target_utilization)
    return {"peak_rps": peak_per_second, "worker_slots": slots}


def known_event_count(events, entity, prediction_time, window):
    """Times are comparable numbers; events have stable IDs and one row per event."""
    if window <= 0:
        raise ValueError("window must be positive")
    return sum(
        event["entity"] == entity
        and prediction_time - window < event["event_time"] <= prediction_time
        and event["available_time"] <= prediction_time
        for event in events
    )


def canary_assignment(entity_id, release_id, percent):
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    key = f"{release_id}:{entity_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10000
    return "canary" if bucket < percent * 100 else "control"


async def classify_batch(
    items, classify, labels, concurrency=2, retries=1, timeout_seconds=1
):
    """Execution demo, not a durable queue or a requests-per-second limiter.

    A production service persists work and results and acquires a provider-wide
    request/token rate permit before each attempt. This bounds only in-flight calls.
    """
    if concurrency < 1 or retries < 0 or timeout_seconds <= 0:
        raise ValueError("invalid execution budget")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("item IDs must be unique")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item):
        for attempt in range(retries + 1):
            try:
                async with semaphore:
                    output = await asyncio.wait_for(
                        classify(item["text"]), timeout_seconds
                    )
                if (
                    not isinstance(output, dict)
                    or set(output) != {"label"}
                    or not isinstance(output["label"], str)
                    or output["label"] not in labels
                ):
                    return item["id"], {"status": "invalid_output", "result": None}
                return item["id"], {"status": "completed", "result": output}
            except (OSError, TimeoutError, asyncio.TimeoutError):
                if attempt == retries:
                    return item["id"], {"status": "retry_exhausted", "result": None}
                # A real service uses bounded exponential backoff with jitter.
                await asyncio.sleep(0)
        raise AssertionError("unreachable")

    return dict(await asyncio.gather(*(one(item) for item in items)))
