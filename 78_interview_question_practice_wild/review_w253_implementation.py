"""Per-node count logic. The transport supplies send_async and calls retry_pending."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Message:
    kind: str
    request_id: str
    payload: object = None

class Node:
    def __init__(self, node_id, parent_id, children):
        self.node_id, self.parent_id = node_id, parent_id
        self.children = tuple(children)
        self.requests = {}
        self.printed = set()

    def send_async(self, to_node_id, message):
        raise NotImplementedError('provided transport')

    def call(self, from_node_id, message):
        rid = message.request_id
        if message.kind == 'request_count':
            if from_node_id != self.parent_id:
                raise ValueError('count request must come from parent/root trigger')
            if rid not in self.requests:
                self.requests[rid] = {'replies': {}, 'total': None}
            state = self.requests[rid]
            if state['total'] is not None:
                self._return(rid)
            else:
                for child in self.children:
                    if child not in state['replies']:
                        self.send_async(child, Message('request_count', rid))
                self._finish(rid)
        elif message.kind == 'count':
            if from_node_id not in self.children or rid not in self.requests:
                return
            value = message.payload
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('subtree count must be a positive integer')
            state = self.requests[rid]
            prior = state['replies'].get(from_node_id)
            if prior is not None and prior != value:
                raise ValueError('inconsistent duplicate count')
            state['replies'][from_node_id] = value
            self._finish(rid)

    def _finish(self, rid):
        state = self.requests[rid]
        if state['total'] is None and len(state['replies']) == len(self.children):
            state['total'] = 1 + sum(state['replies'].values())
            self._return(rid)

    def _return(self, rid):
        total = self.requests[rid]['total']
        if self.parent_id is None:
            if rid not in self.printed:
                self.printed.add(rid)
                print(total)
        else:
            self.send_async(self.parent_id, Message('count', rid, total))

    def retry_pending(self):
        # Call from a timer, with backoff/jitter supplied by the runtime.
        for rid, state in list(self.requests.items()):
            if state['total'] is None:
                for child in self.children:
                    if child not in state['replies']:
                        self.send_async(child, Message('request_count', rid))
