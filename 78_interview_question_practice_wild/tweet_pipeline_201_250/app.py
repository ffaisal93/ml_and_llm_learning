"""W218 runnable local SQLite pipeline. Demo LLM is opt-in via MOCK_LLM=1."""
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DB = os.environ.get('DB_PATH', '/data/tweets.db')
THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '.7'))
LEASE = 60

class Tweet(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=1000)
    author: str = Field(min_length=1)
    created_at: datetime

    @field_validator('text', 'author')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('blank text/author')
        return value

class Insight(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    sentiment: Literal['positive', 'neutral', 'negative']
    topic: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)

    @field_validator('topic')
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError('blank topic')
        return value

def connect():
    db = sqlite3.connect(DB, timeout=30, isolation_level=None)
    db.execute('PRAGMA busy_timeout=30000')
    db.execute('PRAGMA synchronous=FULL')
    db.row_factory = sqlite3.Row
    return db

def init():
    os.makedirs(os.path.dirname(os.path.abspath(DB)), exist_ok=True)
    with connect() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS tweets (
          id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
          attempts INTEGER NOT NULL DEFAULT 0, lease_until REAL, fence INTEGER NOT NULL DEFAULT 0,
          next_try REAL NOT NULL DEFAULT 0, error TEXT);
        CREATE INDEX IF NOT EXISTS pending_idx ON tweets(status,next_try);
        CREATE TABLE IF NOT EXISTS insights (
          tweet_id TEXT PRIMARY KEY REFERENCES tweets(id), sentiment TEXT, topic TEXT, confidence REAL);
        CREATE TABLE IF NOT EXISTS rejects (
          source TEXT, line INTEGER, raw TEXT, reason TEXT, PRIMARY KEY(source,line));
        CREATE TABLE IF NOT EXISTS limits (name TEXT PRIMARY KEY, next_at REAL NOT NULL);
        ''')

def acquire(name, per_second):
    """Database-coordinated leaky admissions, independent by name, no bursts."""
    while True:
        with connect() as db:
            db.execute('BEGIN IMMEDIATE')
            now = time.time()
            row = db.execute('SELECT next_at FROM limits WHERE name=?', (name,)).fetchone()
            due = row[0] if row else now
            if due <= now:
                db.execute('INSERT INTO limits VALUES (?,?) ON CONFLICT(name) DO UPDATE SET next_at=excluded.next_at',
                           (name, now + 1 / per_second))
                db.commit(); return
            db.rollback()
        time.sleep(min(due - now, .1))

def records(path):
    with open(path, encoding='utf-8', newline='') as stream:
        if path.endswith('.csv'):
            yield from enumerate(csv.DictReader(stream), 2)
        elif path.endswith('.jsonl'):
            for i, line in enumerate(stream, 1):
                try: yield i, json.loads(line)
                except json.JSONDecodeError: yield i, {'_malformed_json': line}
        else:
            data = json.load(stream)
            if not isinstance(data, list): raise ValueError('JSON input must be a list')
            yield from enumerate(data, 1)

def ingest(path):
    valid, rejected = 0, 0
    digest = hashlib.sha256()
    with open(path, 'rb') as raw_file:
        for block in iter(lambda: raw_file.read(1024 * 1024), b''):
            digest.update(block)
    source = digest.hexdigest()
    for line, raw in records(path):
        acquire('inbound', 100)
        try:
            tweet = Tweet.model_validate(raw)
        except ValidationError as exc:
            rejected += 1
            with connect() as db:
                db.execute('INSERT OR REPLACE INTO rejects VALUES (?,?,?,?)',
                           (source,line,json.dumps(raw),str(exc)))
            continue
        with connect() as db:
            cur = db.execute('INSERT OR IGNORE INTO tweets(id,payload) VALUES (?,?)',
                             (tweet.id, tweet.model_dump_json()))
            valid += cur.rowcount
    return {'inserted': valid, 'rejected_this_scan': rejected}

def claim():
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        now = time.time()
        db.execute("UPDATE tweets SET status='pending' WHERE status='processing' AND lease_until < ?", (now,))
        row = db.execute("SELECT * FROM tweets WHERE status='pending' AND next_try<=? ORDER BY id LIMIT 1", (now,)).fetchone()
        if not row:
            db.commit(); return None
        db.execute("UPDATE tweets SET status='processing', attempts=attempts+1, fence=fence+1, lease_until=? WHERE id=?",
                   (now+LEASE,row['id']))
        fresh = db.execute('SELECT * FROM tweets WHERE id=?', (row['id'],)).fetchone()
        db.commit(); return dict(fresh)

def classify(tweet):
    if os.environ.get('MOCK_LLM') == '1':
        return {'sentiment': 'neutral', 'topic': 'demo', 'confidence': .95}
    endpoint, secret = os.environ['LLM_URL'], os.environ['LLM_API_KEY']
    # Adapter contract: POST a tweet; provider returns exactly Insight JSON.
    request = urllib.request.Request(endpoint, data=tweet['payload'].encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer '+secret,
        'Idempotency-Key': 'tweet-insight-v1:'+tweet['id']})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)

def process_one(call=classify):
    row = claim()
    if row is None: return False
    try:
        acquire('outbound', 10)  # retries enter through here too
        insight = Insight.model_validate(call(row))
        if insight.confidence < THRESHOLD:
            raise ValueError('confidence below threshold')
        with connect() as db:
            db.execute('BEGIN IMMEDIATE')
            owner = db.execute("SELECT 1 FROM tweets WHERE id=? AND fence=? AND status='processing'",
                               (row['id'],row['fence'])).fetchone()
            if owner:
                db.execute('INSERT OR REPLACE INTO insights VALUES (?,?,?,?)',
                           (row['id'],insight.sentiment,insight.topic,insight.confidence))
                db.execute("UPDATE tweets SET status='done',error=NULL,lease_until=NULL WHERE id=?", (row['id'],))
            db.commit()
    except Exception as exc:
        attempts = row['attempts']
        # All failures are bounded; 429/5xx and parse failures share rate control.
        delay = min(60, 2 ** attempts) + random.random()
        if isinstance(exc, urllib.error.HTTPError):
            try: delay = max(delay, float(exc.headers.get('Retry-After', 0)))
            except ValueError: pass
        status = 'failed' if attempts >= 3 else 'pending'
        with connect() as db:
            db.execute("UPDATE tweets SET status=?,next_try=?,lease_until=NULL,error=? WHERE id=? AND fence=? AND status='processing'",
                       (status,time.time()+delay,type(exc).__name__+': '+str(exc)[:500],row['id'],row['fence']))
    return True

def worker():
    while True:
        if not process_one(): time.sleep(.1)

if __name__ == '__main__':
    init()
    if sys.argv[1] == 'ingest': print(json.dumps(ingest(sys.argv[2])))
    elif sys.argv[1] == 'worker': worker()
    elif sys.argv[1] != 'init': raise SystemExit('init | ingest FILE | worker')
