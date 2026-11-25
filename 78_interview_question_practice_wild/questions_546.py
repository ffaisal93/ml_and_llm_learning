"""W546 runnable standard-library REST demo. python questions_546.py"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import json
import threading

class Store:
    def __init__(self):
        self.questions={};self.votes={};self.next_id=1;self.lock=threading.RLock()
    def create(self,title,body,author):
        if not isinstance(title,str) or not title.strip() or not isinstance(body,str):
            raise ValueError('Need nonempty title and string body')
        if len(title)>300 or len(body)>20000:raise ValueError('Question too long')
        with self.lock:
            identifier=self.next_id;self.next_id+=1
            self.questions[identifier]=dict(id=identifier,title=title.strip(),body=body,author=author)
            return self.get(identifier)
    def get(self,identifier):
        with self.lock:
            question=self.questions[identifier].copy()
            question['score']=sum(v for (qid,user),v in self.votes.items() if qid==identifier)
            return question
    def search(self,query='',offset=0,limit=20):
        with self.lock:
            ids=[i for i,q in self.questions.items()
                 if query.casefold() in (q['title']+' '+q['body']).casefold()]
            return dict(items=[self.get(i) for i in ids[offset:offset+limit]],total=len(ids))
    def vote(self,identifier,user,value):
        if type(value) is not int or value not in (-1,0,1):raise ValueError('Vote must be -1, 0 or 1')
        with self.lock:
            if identifier not in self.questions:raise KeyError(identifier)
            self.votes[identifier,user]=value  # set/replace rather than increment
            return self.get(identifier)

def handler_for(store):
 class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_json(self,status,payload):
        body=json.dumps(payload).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):self.dispatch()
    def do_POST(self):self.dispatch()
    def do_PUT(self):self.dispatch()
    def dispatch(self):
        try:
            parsed=urlparse(self.path);parts=parsed.path.strip('/').split('/')
            data={}
            if self.command in ('POST','PUT'):
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536:
                    self.send_json(413,{'error':'Body must be 1..65536 bytes'});return
                data=json.loads(self.rfile.read(size))
                if not isinstance(data,dict):raise ValueError('Expected JSON object')
            if self.command=='GET' and parts==['questions']:
                query=parse_qs(parsed.query)
                offset=int(query.get('offset',['0'])[0]);limit=int(query.get('limit',['20'])[0])
                if offset<0 or not 1<=limit<=100:raise ValueError('Invalid pagination')
                self.send_json(200,store.search(query.get('q',[''])[0],offset,limit));return
            if self.command=='GET' and len(parts)==2 and parts[0]=='questions':
                self.send_json(200,store.get(int(parts[1])));return
            # DEMO identity only: callers can forge this header. Replace with real auth in deployment.
            user=self.headers.get('X-Demo-User')
            if not user:
                self.send_json(401,{'error':'X-Demo-User is required for this local demo'});return
            if self.command=='POST' and parts==['questions']:
                if set(data)!={'title','body'}:raise ValueError('Expected title and body only')
                self.send_json(201,store.create(data['title'],data['body'],user));return
            if self.command=='PUT' and len(parts)==3 and parts[0]=='questions' and parts[2]=='vote':
                if set(data)!={'value'}:raise ValueError('Expected value only')
                self.send_json(200,store.vote(int(parts[1]),user,data['value']));return
            self.send_json(404,{'error':'Route not found'})
        except KeyError:self.send_json(404,{'error':'Question not found'})
        except (ValueError,UnicodeDecodeError) as error:self.send_json(400,{'error':str(error)})
 return Handler

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',8080),handler_for(Store()))
    print('Local demo at http://127.0.0.1:8080; data resets on restart')
    server.serve_forever()
