"""Local TCP chat demo with SQLite identity/memberships/history and bounded writers."""

import argparse
import asyncio
import hashlib
import hmac
import re
import secrets
import sqlite3
import uuid


class Session:
    def __init__(self, writer):
        self.writer = writer
        self.queue = asyncio.Queue(maxsize=100)
        self.user = None
        self.nick = None
        self.rooms = set()
        self.current = None

    def emit(self, text):
        try:
            self.queue.put_nowait(text + "\n")
        except asyncio.QueueFull:
            self.writer.close()

    async def write_loop(self):
        while True:
            self.writer.write((await self.queue.get()).encode())
            await asyncio.wait_for(self.writer.drain(), 2.0)


class ChatServer:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,token_hash TEXT NOT NULL,nick TEXT UNIQUE NOT NULL);
        CREATE TABLE IF NOT EXISTS rooms(name TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS memberships(user_id TEXT,room TEXT,PRIMARY KEY(user_id,room));
        CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,room TEXT,user_id TEXT,nick TEXT,body TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE INDEX IF NOT EXISTS history ON messages(room,id DESC);
        """)
        self.sessions = set()

    @staticmethod
    def valid_name(name):
        return re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name) is not None

    def history(self, session, room):
        rows = self.db.execute(
            "SELECT nick,body FROM messages WHERE room=? ORDER BY id DESC LIMIT 10",
            (room,),
        ).fetchall()
        session.emit(f"HISTORY {room}")
        for nick, body in reversed(rows):
            session.emit(f"[{room}] {nick}: {body}")
        session.emit(f"END_HISTORY {room}")

    def dispatch(self, s, line):
        if not line:
            return True
        command, *rest = line.split(maxsplit=1)
        arg = rest[0] if rest else ""
        if line.startswith("/"):
            if command == "/help":
                s.emit(
                    "/help /nick NAME /login ID TOKEN /users /create ROOM /join ROOM /rooms /quit"
                )
                return True
            if command == "/quit":
                s.emit("BYE")
                return False
            if command == "/nick":
                if not self.valid_name(arg):
                    s.emit("ERROR nickname must be 1-32 letters/digits/_/-")
                    return True
                try:
                    with self.db:
                        if s.user:
                            self.db.execute(
                                "UPDATE users SET nick=? WHERE id=?", (arg, s.user)
                            )
                            token = None
                        else:
                            ident = uuid.uuid4().hex
                            token = secrets.token_urlsafe(32)
                            self.db.execute(
                                "INSERT INTO users VALUES(?,?,?)",
                                (
                                    ident,
                                    hashlib.sha256(token.encode()).hexdigest(),
                                    arg,
                                ),
                            )
                            s.user = ident
                    s.nick = arg
                    for other in self.sessions:
                        if other.user == s.user:
                            other.nick = arg
                    s.emit(
                        f"id={s.user} token={token} nick={arg}"
                        if token
                        else f"nick={arg}"
                    )
                except sqlite3.IntegrityError:
                    s.emit("ERROR nickname already used")
                return True
            if command == "/login":
                if s.user:
                    s.emit("ERROR already authenticated")
                    return True
                fields = arg.split()
                if len(fields) != 2:
                    s.emit("ERROR /login ID TOKEN")
                    return True
                row = self.db.execute(
                    "SELECT token_hash,nick FROM users WHERE id=?", (fields[0],)
                ).fetchone()
                if row is None or not hmac.compare_digest(
                    row[0], hashlib.sha256(fields[1].encode()).hexdigest()
                ):
                    s.emit("ERROR login failed")
                    return True
                s.user, s.nick = fields[0], row[1]
                s.rooms = {
                    r[0]
                    for r in self.db.execute(
                        "SELECT room FROM memberships WHERE user_id=?", (s.user,)
                    )
                }
                s.current = sorted(s.rooms)[0] if s.rooms else None
                s.emit(f"welcome back {s.nick}; current={s.current}")
                for room in sorted(s.rooms):
                    self.history(s, room)
                return True
            if not s.user:
                s.emit("ERROR identify with /nick NAME or /login ID TOKEN")
                return True
            if command == "/users":
                s.emit(
                    "USERS "
                    + " ".join(sorted({p.nick for p in self.sessions if p.user}))
                )
            elif command == "/rooms":
                s.emit(
                    "ROOMS "
                    + " ".join(
                        r[0]
                        for r in self.db.execute("SELECT name FROM rooms ORDER BY name")
                    )
                )
            elif command == "/create":
                if not self.valid_name(arg):
                    s.emit("ERROR invalid room name")
                    return True
                with self.db:
                    self.db.execute("INSERT OR IGNORE INTO rooms VALUES(?)", (arg,))
                s.emit(f"ROOM {arg}")
            elif command == "/join":
                if (
                    self.db.execute(
                        "SELECT 1 FROM rooms WHERE name=?", (arg,)
                    ).fetchone()
                    is None
                ):
                    s.emit("ERROR no such room")
                    return True
                with self.db:
                    self.db.execute(
                        "INSERT OR IGNORE INTO memberships VALUES(?,?)", (s.user, arg)
                    )
                s.rooms.add(arg)
                s.current = arg
                self.history(s, arg)
            else:
                s.emit("ERROR unknown command; use /help")
            return True
        if not s.user:
            s.emit("ERROR identify with /nick NAME or /login ID TOKEN")
            return True
        if s.current is None:
            s.emit("ERROR join a room using /join ROOM")
            return True
        if len(line) > 4096:
            s.emit("ERROR message too long")
            return True
        with self.db:
            self.db.execute(
                "INSERT INTO messages(room,user_id,nick,body) VALUES(?,?,?,?)",
                (s.current, s.user, s.nick, line),
            )
            self.db.execute(
                "DELETE FROM messages WHERE room=? AND id NOT IN (SELECT id FROM messages WHERE room=? ORDER BY id DESC LIMIT 1000)",
                (s.current, s.current),
            )
        for peer in list(self.sessions):
            if s.current in peer.rooms:
                peer.emit(f"[{s.current}] {s.nick}: {line}")
        return True

    async def handle(self, reader, writer):
        s = Session(writer)
        self.sessions.add(s)
        output = asyncio.create_task(s.write_loop())
        output.add_done_callback(
            lambda task: (
                writer.close()
                if not task.cancelled() and task.exception() is not None
                else None
            )
        )
        s.emit("WELCOME /nick NAME or /login ID TOKEN; /help")
        try:
            while not writer.is_closing():
                data = await reader.readline()
                if not data:
                    break
                try:
                    line = data.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError:
                    s.emit("ERROR invalid UTF-8")
                    continue
                try:
                    keep = self.dispatch(s, line)
                except sqlite3.Error:
                    s.emit("ERROR storage unavailable; not saved")
                    continue
                if not keep:
                    await asyncio.sleep(0.02)
                    break
        except (ValueError, ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self.sessions.discard(s)
            output.cancel()
            try:
                await output
            except (asyncio.CancelledError, Exception):
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass


async def run(port, path):
    chat = ChatServer(path)
    server = await asyncio.start_server(chat.handle, "127.0.0.1", port, limit=8192)
    try:
        async with server:
            await server.serve_forever()
    finally:
        chat.db.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("port", type=int)
    p.add_argument("--db", default="chat_demo.sqlite3")
    a = p.parse_args()
    try:
        asyncio.run(run(a.port, a.db))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
