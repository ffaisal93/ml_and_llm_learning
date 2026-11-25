"""W320 Flask/Redis demo; SQLite read path can be swapped for a pooled DB."""
import hashlib
import json
import sqlite3
from flask import Flask,request,jsonify
from itsdangerous import URLSafeSerializer,BadSignature

LIMIT_LUA="""
local n=redis.call('INCR',KEYS[1])
if n==1 then redis.call('PEXPIRE',KEYS[1],60000) end
return {n,redis.call('PTTL',KEYS[1])}
"""

def create_app(db_path,redis_client,tokens,signing_secret):
    app=Flask(__name__)
    signer=URLSafeSerializer(signing_secret,salt='data-cursor-v1')

    @app.get('/entries')
    def entries():
        bearer=request.headers.get('Authorization','')
        tenant=tokens.get(bearer.removeprefix('Bearer ')) if bearer.startswith('Bearer ') else None
        if tenant is None:return jsonify(error='unauthorized'),401
        try:
            used,ttl=redis_client.eval(LIMIT_LUA,1,'rl:'+tenant)
        except Exception:
            return jsonify(error='limiter_unavailable'),503
        if int(used)>60:
            return jsonify(error='rate_limited'),429,{'Retry-After':str(max(1,(int(ttl)+999)//1000))}
        try:
            limit=int(request.args.get('limit','100'))
            if not 1<=limit<=100:raise ValueError()
            after=0
            if request.args.get('cursor'):
                payload=signer.loads(request.args['cursor'])
                if payload['tenant']!=tenant:raise ValueError()
                after=int(payload['after'])
                if after<0:raise ValueError()
        except (ValueError,KeyError,TypeError,BadSignature):
            return jsonify(error='invalid_pagination'),400
        key='entries:v1:'+hashlib.sha256(json.dumps([tenant,after,limit]).encode()).hexdigest()
        try:cached=redis_client.get(key)
        except Exception:cached=None
        if cached is not None:return app.response_class(cached,mimetype='application/json')
        # Demo schema entries(id INTEGER PRIMARY KEY,tenant TEXT,payload TEXT),
        # index CREATE INDEX entries_tenant_id ON entries(tenant,id).
        with sqlite3.connect(db_path,timeout=2) as db:
            db.row_factory=sqlite3.Row
            rows=db.execute('SELECT id,payload FROM entries WHERE tenant=? AND id>? ORDER BY id LIMIT ?',
                            (tenant,after,limit+1)).fetchall()
        has_more=len(rows)>limit;rows=rows[:limit]
        result={'items':[dict(r) for r in rows],
                'next_cursor':signer.dumps({'tenant':tenant,'after':rows[-1]['id']}) if has_more else None}
        body=json.dumps(result)
        try:redis_client.setex(key,30,body)
        except Exception:pass
        return app.response_class(body,mimetype='application/json')
    return app

if __name__=='__main__':
    import os,redis
    app=create_app(os.environ['DB_PATH'],redis.Redis.from_url(os.environ['REDIS_URL']),
                   {os.environ['DEMO_API_TOKEN']:'demo'},os.environ['CURSOR_SECRET'])
    app.run(host='127.0.0.1',port=5000)
