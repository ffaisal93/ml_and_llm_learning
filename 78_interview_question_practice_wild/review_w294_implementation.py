"""Thread-safe snapshot publication; no persistence is claimed."""
import threading

class Broker:
    def __init__(self):
        self._lock = threading.Lock()
        self._topics = {}
        self._sequence = 0

    def subscribe(self, subscriber, topic, priority=0):
        with self._lock:
            entries = self._topics.setdefault(topic, {})
            if subscriber in entries:
                _, order = entries[subscriber]
                entries[subscriber] = (priority, order)
            else:
                self._sequence += 1
                entries[subscriber] = (priority, self._sequence)

    def unsubscribe(self, subscriber, topic):
        with self._lock:
            entries = self._topics.get(topic)
            if entries is not None:
                entries.pop(subscriber, None)
                if not entries:
                    self._topics.pop(topic, None)

    def publish(self, topic, message):
        with self._lock:
            snapshot = list(self._topics.get(topic, {}).items())
        snapshot.sort(key=lambda pair: (-pair[1][0], pair[1][1]))
        errors = []
        for subscriber, _ in snapshot:
            try:
                subscriber(message)
            except Exception as exc:
                errors.append((subscriber, f'{type(exc).__name__}: {exc}'))
        return errors
