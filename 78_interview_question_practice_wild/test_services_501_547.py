import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from questions_546 import Store,handler_for
from course_547 import create_app,Course,Chapter

class RestTests(unittest.TestCase):
 def test_question_routes_and_idempotent_votes(self):
  server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(Store()))
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  base=f'http://127.0.0.1:{server.server_port}'
  def call(path,method='GET',data=None,user=None):
   headers={'Content-Type':'application/json'}
   if user:headers['X-Demo-User']=user
   req=Request(base+path,data=None if data is None else json.dumps(data).encode(),method=method,headers=headers)
   try:
    with urlopen(req,timeout=3) as response:return response.status,json.load(response)
   except HTTPError as error:return error.code,json.load(error)
  try:
   self.assertEqual(call('/questions','POST',{'title':'T','body':'B'})[0],401)
   status,q=call('/questions','POST',{'title':'Softmax','body':'Why shift?'},'a');self.assertEqual(status,201)
   for _ in range(2):self.assertEqual(call('/questions/1/vote','PUT',{'value':1},'b')[1]['score'],1)
   self.assertEqual(call('/questions/1/vote','PUT',{'value':-1},'b')[1]['score'],-1)
   self.assertEqual(call('/questions?q=soft')[1]['total'],1)
   self.assertEqual(call('/questions/999')[0],404)
   self.assertEqual(call('/questions/1/vote','PUT',{'value':2},'b')[0],400)
  finally:server.shutdown();server.server_close();thread.join()
 def test_course_end_to_end(self):
  with tempfile.TemporaryDirectory() as directory:
   app=create_app('sqlite:///'+str(Path(directory)/'test.db'),'test-secret')
   DB=app.extensions['course_db']
   with DB.begin() as db:
    db.add(Course(id=1,title='Machine Learning',category='ML',introduction='Foundations'))
    db.flush();db.add(Chapter(course_id=1,position=1,title='Loss',body='Cross entropy'))
   client=app.test_client()
   token=client.get('/api/csrf').json['csrf'];headers={'X-CSRF-Token':token}
   self.assertEqual(client.post('/api/login',json={},headers={'X-CSRF-Token':'é'}).status_code,403)
   self.assertEqual(client.post('/api/register',json={'username':'ß'*100,'password':'strong-pass'},headers=headers).status_code,400)
   self.assertEqual(client.post('/api/register',json={'username':'u','password':'strong-pass'},headers=headers).status_code,201)
   self.assertEqual(client.post('/api/register',json={'username':'u','password':'strong-pass'},headers=headers).status_code,409)
   self.assertEqual(client.put('/api/favorites/1',headers=headers).status_code,401)
   login=client.post('/api/login',json={'username':'u','password':'strong-pass'},headers=headers)
   self.assertEqual(login.status_code,200)
   self.assertIn('HttpOnly',login.headers['Set-Cookie'])
   self.assertEqual(client.put('/api/favorites/1',headers=headers).status_code,403)
   headers={'X-CSRF-Token':login.json['csrf']}
   self.assertEqual(client.get('/api/courses?q=machine&category=ML').json['items'][0]['id'],1)
   self.assertEqual(client.get('/api/courses/1').json['chapters'][0]['body'],'Cross entropy')
   for _ in range(2):self.assertEqual(client.put('/api/favorites/1',headers=headers).status_code,204)
   self.assertEqual(len(client.get('/api/favorites').json['items']),1)
   self.assertEqual(client.delete('/api/favorites/1',headers=headers).status_code,204)
   self.assertEqual(client.get('/api/favorites').json['items'],[])
   self.assertEqual(client.post('/api/logout',headers=headers).status_code,204)
   self.assertEqual(client.get('/api/favorites').status_code,401)

if __name__=='__main__':unittest.main()
