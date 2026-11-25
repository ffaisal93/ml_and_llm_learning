"""W219 demonstration adapter: provider contract must be mapped explicitly."""

import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request


class SpacedLimiter:
    def __init__(self, rate):
        self.period, self.next_at = 1 / rate, 0.0

    def wait(self):
        delay = self.next_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_at = time.monotonic() + self.period


def fetch_http(keyword, cursor):
    # This is an INTERNAL adapter endpoint contract, not a claim about a live API.
    params = urllib.parse.urlencode({"keyword": keyword, "cursor": cursor or ""})
    request = urllib.request.Request(
        os.environ["API_URL"] + "?" + params,
        headers={"Authorization": "Bearer " + os.environ["API_TOKEN"]},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def analyze(keyword, fetch_page, db_path="engagement.db", inbound=None, outbound=None):
    """Page contract: {items:[{id,text,likes,retweets,comments}], next_cursor}."""
    inbound = inbound or SpacedLimiter(100)
    outbound = outbound or SpacedLimiter(10)
    inbound.wait()  # each accepted search, not each API page
    db = sqlite3.connect(db_path)
    db.executescript(
        """CREATE TABLE IF NOT EXISTS engagement(
       keyword TEXT,id TEXT,text TEXT,likes INTEGER,retweets INTEGER,comments INTEGER,
       PRIMARY KEY(keyword,id));
       CREATE TABLE IF NOT EXISTS checkpoints(keyword TEXT PRIMARY KEY,cursor TEXT,done INTEGER);"""
    )
    row = db.execute(
        "SELECT cursor,done FROM checkpoints WHERE keyword=?", (keyword,)
    ).fetchone()
    cursor, done = (row[0], bool(row[1])) if row else (None, False)
    while not done:
        for attempt in range(4):
            outbound.wait()  # every attempt spends from the outbound rate budget
            try:
                page = fetch_page(keyword, cursor)
                break
            except Exception:
                if attempt == 3:
                    db.close()
                    raise
                time.sleep(min(8, 2**attempt))
        next_cursor = page.get("next_cursor")
        if next_cursor is not None and next_cursor == cursor:
            db.close()
            raise ValueError("non-advancing cursor")
        with db:
            for item in page["items"]:
                counts = [item[k] for k in ("likes", "retweets", "comments")]
                if any(not isinstance(v, int) or v < 0 for v in counts):
                    raise ValueError("invalid engagement count")
                db.execute(
                    "INSERT OR REPLACE INTO engagement VALUES (?,?,?,?,?,?)",
                    (keyword, str(item["id"]), item["text"], *counts),
                )
            done = next_cursor is None
            db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?)",
                (keyword, next_cursor, int(done)),
            )
        cursor = next_cursor
    # A DB cursor streams rows rather than materializing every tweet in Python.
    cur = db.cursor()
    cur.execute(
        "SELECT COUNT(*),COALESCE(SUM(likes),0),COALESCE(SUM(retweets),0),COALESCE(SUM(comments),0) FROM engagement WHERE keyword=?",
        (keyword,),
    )
    n, likes, retweets, comments = cur.fetchone()
    db.close()
    return dict(
        tweets=n,
        likes=likes,
        retweets=retweets,
        comments=comments,
        engagement_per_tweet=(likes + retweets + comments) / n if n else 0.0,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            analyze(
                os.environ.get("KEYWORD", "demo"),
                fetch_http,
                os.environ.get("DB_PATH", "/data/engagement.db"),
            )
        )
    )
