"""Small deterministic examples, not a distributed scheduler implementation."""

from threading import Lock
import math


class TokenBucket:
    def __init__(self, capacity, refill_per_second, start_time=0.0):
        if not all(math.isfinite(v) for v in (capacity, refill_per_second, start_time)):
            raise ValueError("bucket values must be finite")
        if capacity <= 0 or refill_per_second <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.tokens = capacity
        self.last_time = start_time
        self.lock = Lock()

    def allow(self, cost, now):
        # now comes from a trusted monotonic server clock, not a client field.
        if not math.isfinite(cost) or not math.isfinite(now) or cost <= 0:
            raise ValueError("need finite time and a positive finite cost")
        with self.lock:
            if now < self.last_time:
                raise ValueError("clock moved backward")
            elapsed = now - self.last_time
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.refill_per_second
            )
            self.last_time = now
            if self.tokens < cost:
                return False
            self.tokens -= cost
            return True


def ready_jobs(dependencies, states):
    """Readiness snapshot for a prevalidated DAG; claiming work is a separate transaction."""
    for parents in dependencies.values():
        if not set(parents) <= dependencies.keys():
            raise ValueError("dependency references an unknown job")
    return sorted(
        job
        for job, parents in dependencies.items()
        if states.get(job) == "PENDING"
        and all(states.get(parent) == "SUCCEEDED" for parent in parents)
    )
