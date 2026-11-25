from collections import deque


class DynamicBatcher:
    """Pure scheduling core; no threads, clock or production retry policy."""
    def __init__(self, model, b_max, t_max):
        if b_max < 1 or t_max < 0:
            raise ValueError("invalid batching limits")
        self.model, self.b_max, self.t_max = model, b_max, t_max
        self.queue, self.next_id = deque(), 0

    def submit(self, x, now):
        request_id = self.next_id
        self.next_id += 1
        self.queue.append((request_id, x, now))
        return request_id

    def next_deadline(self):
        return None if not self.queue else self.queue[0][2] + self.t_max

    def should_dispatch(self, now):
        return bool(self.queue) and (len(self.queue) >= self.b_max or now >= self.next_deadline())

    def dispatch(self, now):
        if not self.should_dispatch(now):
            return []
        batch = [self.queue.popleft() for _ in range(min(self.b_max, len(self.queue)))]
        outputs = list(self.model([item[1] for item in batch]))
        if len(outputs) != len(batch):
            raise ValueError("model returned the wrong number of outputs")
        return [(item[0], result) for item, result in zip(batch, outputs)]

record=[]
def model(xs):record.append(xs);return [x*10 for x in xs]
b=DynamicBatcher(model,3,.05)
assert b.next_deadline() is None and b.dispatch(0)==[]
for x,t in [(1,0),(2,.01),(3,.02)]:b.submit(x,t)
assert b.dispatch(.02)==[(0,10),(1,20),(2,30)] and record==[[1,2,3]]
b=DynamicBatcher(model,3,.05);b.submit(1,0);b.submit(2,.01)
assert b.dispatch(.04)==[] and b.next_deadline()==.05
assert b.dispatch(.05)==[(0,10),(1,20)]
b=DynamicBatcher(model,3,.05)
for x in range(5):b.submit(x,0)
assert len(b.dispatch(.1))==3 and len(b.dispatch(.1))==2
assert DynamicBatcher(model,1,0).dispatch(0)==[]
print('W027: exact FIFO, fill/deadline/backlog examples and output mapping passed')
