import heapq
from collections import deque


class PriorityCrawlerScheduler:
    def __init__(self, priorities):
        self.priorities = dict(priorities)
        self.queues = {d: deque() for d in priorities}
        self.next_allowed = {d: float("-inf") for d in priorities}
        self.state = {d: "idle" for d in priorities}
        self.ready = []
        self.delayed = []

    def submit(self, domain, url, now):
        if domain not in self.queues:
            raise ValueError("unknown domain")
        self.queues[domain].append(url)
        if self.state[domain] == "idle":
            if self.next_allowed[domain] <= now:
                heapq.heappush(self.ready, (-self.priorities[domain], domain))
                self.state[domain] = "ready"
            else:
                heapq.heappush(self.delayed, (self.next_allowed[domain], domain))
                self.state[domain] = "waiting"

    def take(self, now):
        while self.delayed and self.delayed[0][0] <= now:
            _, domain = heapq.heappop(self.delayed)
            heapq.heappush(self.ready, (-self.priorities[domain], domain))
            self.state[domain] = "ready"
        if not self.ready:
            return None
        _, domain = heapq.heappop(self.ready)
        url = self.queues[domain].popleft()
        self.next_allowed[domain] = now + 1.0
        if self.queues[domain]:
            heapq.heappush(self.delayed, (now + 1.0, domain))
            self.state[domain] = "waiting"
        else:
            self.state[domain] = "idle"
        return domain, url

    def next_wakeup(self):
        return None if not self.delayed else self.delayed[0][0]


if __name__ == "__main__":
    s = PriorityCrawlerScheduler({"domain1": 2, "domain2": 1, "domain3": 3})
    for d in s.priorities:
        s.submit(d, d + "/a", 0)
        s.submit(d, d + "/b", 0)
    assert [s.take(0)[0] for _ in range(3)] == ["domain3", "domain1", "domain2"]
    assert s.take(0.99) is None and s.next_wakeup() == 1.0
    assert s.take(1) == ("domain3", "domain3/b")
    assert s.take(1) == ("domain1", "domain1/b")
    assert s.take(1) == ("domain2", "domain2/b") and s.take(1) is None
    s.submit("domain3", "later", 1.1)
    assert s.take(1.9) is None and s.take(2.0) == ("domain3", "later")
    print(
        "W386: priority among eligible domains and inclusive one-second spacing passed"
    )
