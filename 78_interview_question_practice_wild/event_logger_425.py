"""W425: bounded asynchronous appends; Futures acknowledge fsync group commit."""

from concurrent.futures import Future
import json
import os
from pathlib import Path
import queue
import threading
import time


class EventLogger:
    def __init__(
        self, path, max_batch=100, flush_seconds=0.01, queue_size=10000, sync=os.fsync
    ):
        if max_batch < 1 or flush_seconds < 0 or queue_size < 1:
            raise ValueError("Invalid batching limits")
        self.path = Path(path)
        self.max_batch = max_batch
        self.delay = flush_seconds
        self.sync = sync
        self.queue = queue.Queue(queue_size)
        self.lock = threading.Lock()
        self.closed = False
        self.error = None
        if not self.path.parent.is_dir():
            raise ValueError("Create the durable parent directory first")
        self.file = open(self.path, "a+b")
        # Recovery accepts only a complete JSONL prefix, truncating a torn final line.
        self.file.seek(0)
        valid_end = 0
        self.next_sequence = 1
        while True:
            line = self.file.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            record = json.loads(line)
            if record["sequence"] != self.next_sequence:
                raise ValueError("Corrupt sequence")
            self.next_sequence += 1
            valid_end = self.file.tell()
        self.file.truncate(valid_end)
        self.file.flush()
        os.fsync(self.file.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)  # make a newly created log's directory entry durable
        finally:
            os.close(directory)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def append(self, event):
        # Snapshot/validate JSON at the API boundary; do not retain mutable caller objects.
        payload = json.loads(json.dumps(event, allow_nan=False))
        future = Future()
        future.set_running_or_notify_cancel()  # accepting an append is not cancellable
        with self.lock:
            if self.closed or self.error:
                raise RuntimeError("Logger unavailable") from self.error
            try:
                self.queue.put_nowait((payload, future))
            except queue.Full:
                raise BufferError("Log queue full; event not accepted")
        return future

    def _run(self):
        batch = []
        try:
            stop = False
            while not stop:
                item = self.queue.get()
                if item is None:
                    break
                batch = [item]
                deadline = time.monotonic() + self.delay
                while len(batch) < self.max_batch:
                    try:
                        item = self.queue.get(
                            timeout=max(0, deadline - time.monotonic())
                        )
                    except queue.Empty:
                        break
                    if item is None:
                        stop = True
                        break
                    batch.append(item)
                records = []
                sequences = []
                for payload, future in batch:
                    sequence = self.next_sequence
                    self.next_sequence += 1
                    sequences.append(sequence)
                    records.append(
                        json.dumps(
                            dict(sequence=sequence, event=payload), allow_nan=False
                        ).encode()
                        + b"\n"
                    )
                self.file.write(b"".join(records))
                self.file.flush()
                self.sync(self.file.fileno())
                for (_, future), sequence in zip(batch, sequences):
                    if not future.cancelled():
                        future.set_result(sequence)
                batch = []
        except BaseException as error:
            with self.lock:
                self.error = error
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)
            while True:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None and not item[1].done():
                    item[1].set_exception(error)
        finally:
            self.file.close()

    def close(self):
        with self.lock:
            first = not self.closed
            self.closed = True
        if first:
            while self.worker.is_alive():
                try:
                    self.queue.put(None, timeout=0.05)
                    break
                except queue.Full:
                    pass
        self.worker.join()
        if self.error:
            raise RuntimeError("Durable log write failed") from self.error
