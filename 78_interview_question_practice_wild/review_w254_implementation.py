"""Runnable highest-live-ID membership demo; eventual agreement, not consensus."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

class Cluster:
    def __init__(self, node_id, address, timeout=2.0):
        self.id, self.address, self.timeout = node_id, address, timeout
        self.members = {node_id: address}
        self.seen = {node_id: time.monotonic()}
        self.lock = threading.Lock()

    def state(self):
        with self.lock:
            now = time.monotonic()
            live = [self.id] + [i for i,t in self.seen.items() if now-t < self.timeout]
            return {'id': self.id, 'leader': max(live),
                    'members': {str(i): a for i,a in self.members.items()}}

    def merge(self, data):
        with self.lock:
            for ident,address in data['members'].items():
                ident=int(ident)
                if ident in self.members and self.members[ident] != address:
                    raise ValueError('duplicate node ID has another address')
                self.members[ident]=address

    def contact(self, address):
        req=Request(f'http://{address}/join', data=json.dumps(self.state()).encode(),
                    headers={'Content-Type':'application/json'},method='POST')
        with urlopen(req,timeout=min(.3,self.timeout/3)) as response:
            data=json.load(response)
        self.merge(data)
        # Only a direct response refreshes this peer's health, not gossip.
        with self.lock: self.seen[int(data['id'])]=time.monotonic()

    def loop(self, stop):
        with ThreadPoolExecutor(max_workers=7) as pool:
            while not stop.is_set():
                with self.lock: peers=[a for i,a in self.members.items() if i!=self.id]
                def ping(address):
                    try: self.contact(address)
                    except Exception: pass  # timeout is suspicion; try next tick
                list(pool.map(ping,peers))
                stop.wait(.15)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--id',type=int,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--seed')
    parser.add_argument('--timeout',type=float,default=2.)
    args=parser.parse_args()
    if args.timeout<=0: parser.error('timeout must be positive')
    cluster=Cluster(args.id,f'127.0.0.1:{args.port}',args.timeout)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def reply(self,status,payload):
            body=json.dumps(payload).encode(); self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.reply(200,cluster.state()) if self.path=='/leader' else self.reply(404,{})
        def do_POST(self):
            if self.path!='/join': self.reply(404,{});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536: raise ValueError('bad request size')
                cluster.merge(json.loads(self.rfile.read(size)))
                self.reply(200,cluster.state())
            except (ValueError,KeyError,TypeError) as exc: self.reply(400,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    stop=threading.Event()
    thread=threading.Thread(target=cluster.loop,args=(stop,),daemon=True)
    # Start the HTTP server before the background join retries.
    def seed_join():
        while not stop.is_set():
            try: cluster.contact(args.seed);return
            except Exception: stop.wait(.2)
    thread.start()
    if args.seed: threading.Thread(target=seed_join,daemon=True).start()
    try: server.serve_forever(poll_interval=.1)
    finally: stop.set();server.server_close();thread.join(timeout=1)

if __name__=='__main__': main()
