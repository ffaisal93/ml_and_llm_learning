"""Runnable cores for W454, W480 and W487/497; explicit demonstration contracts."""

# BEGIN W454
import numpy as np
from scipy.stats import binom


def quantile_mc(
    draw, half_width=1000.0, confidence=0.95, seed=0, initial=25000, max_paths=400000
):
    """draw(n,rng) returns iid monetary losses; alpha spending covers planned looks."""
    if half_width <= 0 or not 0 < confidence < 1 or initial < 1 or max_paths < initial:
        raise ValueError("invalid precision/confidence/path budget")
    rng = np.random.default_rng(seed)
    losses = np.empty(0)
    n = initial
    stage = 1
    history = []
    while True:
        losses = np.r_[losses, np.asarray(draw(n - len(losses), rng), dtype=float)]
        if losses.shape != (n,) or not np.isfinite(losses).all():
            raise ValueError("invalid loss sample")
        ordered = np.sort(losses)
        alpha = (1 - confidence) / (2**stage)
        lower_rank = int(binom.ppf(alpha / 2, n, 0.99))
        upper_rank = int(binom.ppf(1 - alpha / 2, n, 0.99)) + 1
        low = ordered[lower_rank - 1] if lower_rank >= 1 else -np.inf
        high = ordered[upper_rank - 1] if upper_rank <= n else np.inf
        estimate = float(np.quantile(ordered, 0.99))
        history.append(
            dict(
                paths=n,
                estimate=estimate,
                low=float(low),
                high=float(high),
                look_alpha=alpha,
            )
        )
        if max(estimate - low, high - estimate) <= half_width:
            return {
                "converged": True,
                "estimate": estimate,
                "interval": (float(low), float(high)),
                "history": history,
            }
        if n == max_paths:
            return {
                "converged": False,
                "estimate": estimate,
                "interval": (float(low), float(high)),
                "history": history,
            }
        n = min(max_paths, n * 2)
        stage += 1


# END W454

# BEGIN W480
import math
from collections import OrderedDict
import heapq
import threading
import time


class LocalTTLCache:
    """Item-count-bounded LRU+TTL core; cluster consistency is a separate design."""

    def __init__(self, capacity, clock=time.monotonic):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("positive integer capacity required")
        self.capacity, self.clock = capacity, clock
        self.data, self.expiry, self.generation = OrderedDict(), [], 0
        self.lock = threading.Lock()

    def _purge(self, now):
        while self.expiry and self.expiry[0][0] <= now:
            _, generation, key = heapq.heappop(self.expiry)
            if key in self.data and self.data[key][2] == generation:
                self.data.pop(key)
        if len(self.expiry) > 2 * len(self.data) + 100:
            self.expiry = [
                (e, g, k) for k, (v, e, g) in self.data.items() if e is not None
            ]
            heapq.heapify(self.expiry)

    def set(self, key, value, ttl=None):
        if ttl is not None and (
            not isinstance(ttl, (float, int))
            or isinstance(ttl, bool)
            or not math.isfinite(ttl)
            or ttl <= 0
        ):
            raise ValueError("positive finite ttl required")
        with self.lock:
            now = self.clock()
            self._purge(now)
            self.generation += 1
            expiry = now + ttl if ttl is not None else None
            self.data.pop(key, None)
            self.data[key] = (value, expiry, self.generation)
            if expiry is not None:
                heapq.heappush(self.expiry, (expiry, self.generation, key))
            while len(self.data) > self.capacity:
                self.data.popitem(last=False)

    def get(self, key):
        with self.lock:
            self._purge(self.clock())
            if key not in self.data:
                raise KeyError(key)
            self.data.move_to_end(key)
            return self.data[key][0]

    def delete(self, key):
        with self.lock:
            self._purge(self.clock())
            return self.data.pop(key, None) is not None


# END W480

# BEGIN W487
import time
import json
import math
import re
import secrets
import sqlite3
import string
from contextlib import closing
from urllib.parse import urlsplit


class AliasTaken(ValueError):
    pass


class LinkExpired(LookupError):
    pass


class LinkNotFound(LookupError):
    pass


class ShortCodeStore:
    """Single SQLite authority demo; unique code is the final collision guard."""

    reserved = {"api", "admin", "login", "static"}

    def __init__(self, path, clock=time.time, generate=None):
        self.path, self.clock = path, clock
        self.generate = generate or (
            lambda: "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(8)
            )
        )
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript(
                """CREATE TABLE IF NOT EXISTS links(code TEXT PRIMARY KEY,url TEXT NOT NULL,expires REAL,disabled INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS requests(key TEXT PRIMARY KEY,payload TEXT NOT NULL,code TEXT NOT NULL);"""
            )

    def create(
        self,
        long_url,
        optional_expiry=None,
        optional_custom_alias=None,
        idempotency_key=None,
    ):
        if (
            not isinstance(long_url, str)
            or len(long_url) > 8192
            or any(c.isspace() or ord(c) < 32 for c in long_url)
        ):
            raise ValueError("invalid URL")
        try:
            parts = urlsplit(long_url)
            port = parts.port
            if (
                parts.scheme not in ("http", "https")
                or not parts.hostname
                or parts.username is not None
                or parts.password is not None
            ):
                raise ValueError("only HTTP(S) URLs without credentials are supported")
        except (ValueError, TypeError):
            raise ValueError("invalid URL")
        if optional_expiry is not None and (
            not isinstance(optional_expiry, (float, int))
            or isinstance(optional_expiry, bool)
            or not math.isfinite(optional_expiry)
        ):
            raise ValueError("expiry must be a future epoch timestamp")
        alias = optional_custom_alias
        if alias is not None and (
            not isinstance(alias, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alias)
            or alias.lower() in self.reserved
        ):
            raise ValueError("invalid/reserved alias")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200
        ):
            raise ValueError("invalid idempotency key")
        payload = json.dumps([long_url, optional_expiry, alias], separators=(",", ":"))
        with closing(
            sqlite3.connect(self.path, timeout=10, isolation_level=None)
        ) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if idempotency_key is not None:
                    old = db.execute(
                        "SELECT payload,code FROM requests WHERE key=?",
                        (idempotency_key,),
                    ).fetchone()
                    if old:
                        if old[0] != payload:
                            raise ValueError(
                                "idempotency key reused with different request"
                            )
                        db.commit()
                        return old[1]
                if optional_expiry is not None and optional_expiry <= self.clock():
                    raise ValueError("expiry must be in the future")
                for _ in range(32):
                    code = alias if alias is not None else self.generate()
                    if (
                        not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", code)
                        or code.lower() in self.reserved
                    ):
                        if alias is not None:
                            raise ValueError("invalid alias")
                        continue
                    try:
                        db.execute(
                            "INSERT INTO links(code,url,expires) VALUES(?,?,?)",
                            (code, long_url, optional_expiry),
                        )
                        break
                    except sqlite3.IntegrityError:
                        if alias is not None:
                            raise AliasTaken(code)
                else:
                    raise RuntimeError("collision retry budget exhausted")
                if idempotency_key is not None:
                    db.execute(
                        "INSERT INTO requests VALUES(?,?,?)",
                        (idempotency_key, payload, code),
                    )
                db.commit()
                return code
            except BaseException:
                db.rollback()
                raise

    def resolve(self, code):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT url,expires,disabled FROM links WHERE code=?", (code,)
            ).fetchone()
        if row is None:
            raise LinkNotFound(code)
        url, expiry, disabled = row
        if disabled or (expiry is not None and self.clock() >= expiry):
            raise LinkExpired(code)
        return url


# END W487

# BEGIN W497
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


def make_fee_schedule(rate):
    """Configured rate; two runtime args: cap_enabled bool and quantity X."""
    rate = Decimal(str(rate))

    def fee(cap_enabled, X):
        if not isinstance(cap_enabled, bool):
            raise ValueError("boolean cap flag required")
        try:
            quantity = Decimal(str(X))
        except InvalidOperation:
            raise ValueError("numeric quantity required")
        if not quantity.is_finite() or quantity < 0:
            raise ValueError("finite nonnegative quantity required")
        total = rate * quantity
        if cap_enabled:
            total = min(total, Decimal("1.50"))
        return total.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)

    return fee


fee_young, fee_middle, fee_old = (make_fee_schedule(r) for r in (".2", ".3", ".4"))
# END W497
