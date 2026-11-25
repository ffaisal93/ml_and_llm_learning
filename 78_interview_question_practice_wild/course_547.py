"""W547 Flask + SQLAlchemy. Set DATABASE_URL to PostgreSQL for deployment."""
import os
import secrets
from flask import Flask,request,session,jsonify
from sqlalchemy import (create_engine,Column,Integer,String,Text,ForeignKey,
                        UniqueConstraint,select,or_,event)
from sqlalchemy.orm import declarative_base,sessionmaker
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash,check_password_hash

Base=declarative_base()
class User(Base):
    __tablename__='users'
    id=Column(Integer,primary_key=True)
    username=Column(String(100),unique=True,nullable=False)
    password_hash=Column(Text,nullable=False)
class Course(Base):
    __tablename__='courses'
    id=Column(Integer,primary_key=True);title=Column(String(300),nullable=False)
    category=Column(String(100),nullable=False,index=True);introduction=Column(Text,nullable=False)
class Chapter(Base):
    __tablename__='chapters'
    id=Column(Integer,primary_key=True);course_id=Column(ForeignKey('courses.id',ondelete='CASCADE'),nullable=False,index=True)
    position=Column(Integer,nullable=False);title=Column(String(300),nullable=False);body=Column(Text,nullable=False)
    __table_args__=(UniqueConstraint('course_id','position'),)
class Favorite(Base):
    __tablename__='favorites'
    user_id=Column(ForeignKey('users.id',ondelete='CASCADE'),primary_key=True)
    course_id=Column(ForeignKey('courses.id',ondelete='CASCADE'),primary_key=True,index=True)

def create_app(database_url,secret_key,secure_cookies=False):
    app=Flask(__name__)
    app.config.update(SECRET_KEY=secret_key,SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',SESSION_COOKIE_SECURE=secure_cookies,MAX_CONTENT_LENGTH=65536)
    engine=create_engine(database_url)
    if engine.dialect.name=='sqlite':
        @event.listens_for(engine,'connect')
        def sqlite_foreign_keys(connection,record):connection.execute('PRAGMA foreign_keys=ON')
    Base.metadata.create_all(engine)  # use reviewed migrations rather than this at production startup
    DB=sessionmaker(engine)
    app.extensions['course_db']=DB
    def current_user(db):
        uid=session.get('user_id')
        return db.get(User,uid) if uid is not None else None
    def body():
        value=request.get_json(silent=True)
        return value if isinstance(value,dict) else {}
    @app.before_request
    def csrf_check():
        # Token required on registration/login too; protects login CSRF.
        if request.method in ('POST','PUT','DELETE','PATCH'):
            supplied=request.headers.get('X-CSRF-Token','')
            expected=session.get('csrf')
            if not expected or not secrets.compare_digest(supplied,expected):
                return jsonify(error='Invalid CSRF token'),403
    @app.get('/api/csrf')
    def csrf():
        session.setdefault('csrf',secrets.token_hex(24))
        return jsonify(csrf=session['csrf'])
    @app.post('/api/register')
    def register():
        data=body();name=data.get('username');password=data.get('password')
        if not isinstance(name,str) or not name.strip() or len(name)>100 or not isinstance(password,str) or not 8<=len(password)<=200:
            return jsonify(error='Need username and password of 8..200 characters'),400
        with DB.begin() as db:
            user=User(username=name.strip().casefold(),password_hash=generate_password_hash(password))
            db.add(user)
            try:db.flush()
            except IntegrityError:
                db.rollback();return jsonify(error='Username unavailable'),409
            uid=user.id
        return jsonify(id=uid),201
    @app.post('/api/login')
    def login():
        data=body();name=data.get('username');password=data.get('password')
        if not isinstance(name,str) or not isinstance(password,str):return jsonify(error='Invalid credentials'),401
        with DB() as db:
            user=db.scalar(select(User).where(User.username==name.strip().casefold()))
            if user is None or not check_password_hash(user.password_hash,password):
                return jsonify(error='Invalid credentials'),401
            uid=user.id
        session.clear();session.update(user_id=uid,csrf=secrets.token_hex(24))
        return jsonify(id=uid,csrf=session['csrf'])
    @app.post('/api/logout')
    def logout():session.clear();return '',204
    @app.get('/api/courses')
    def courses():
        query=request.args.get('q','');category=request.args.get('category')
        try:page=max(0,int(request.args.get('page',0)))
        except ValueError:return jsonify(error='Invalid page'),400
        statement=select(Course)
        if category:statement=statement.where(Course.category==category)
        if query:statement=statement.where(or_(Course.title.icontains(query,autoescape=True),Course.introduction.icontains(query,autoescape=True)))
        with DB() as db:
            rows=db.scalars(statement.order_by(Course.id).offset(page*20).limit(20)).all()
            return jsonify(items=[dict(id=c.id,title=c.title,category=c.category) for c in rows])
    @app.get('/api/courses/<int:course_id>')
    def detail(course_id):
        with DB() as db:
            course=db.get(Course,course_id)
            if course is None:return jsonify(error='Not found'),404
            chapters=db.scalars(select(Chapter).where(Chapter.course_id==course_id).order_by(Chapter.position)).all()
            return jsonify(id=course.id,title=course.title,introduction=course.introduction,
                chapters=[dict(id=c.id,title=c.title,position=c.position,body=c.body) for c in chapters])
    @app.route('/api/favorites/<int:course_id>',methods=['PUT','DELETE'])
    def favorite(course_id):
        with DB.begin() as db:
            user=current_user(db)
            if user is None:return jsonify(error='Login required'),401
            if db.get(Course,course_id) is None:return jsonify(error='Not found'),404
            # Lock user row serializes concurrent favorite changes in PostgreSQL.
            db.scalar(select(User).where(User.id==user.id).with_for_update())
            existing=db.get(Favorite,(user.id,course_id))
            if request.method=='PUT' and existing is None:db.add(Favorite(user_id=user.id,course_id=course_id))
            if request.method=='DELETE' and existing is not None:db.delete(existing)
        return '',204
    @app.get('/api/favorites')
    def favorites():
        with DB() as db:
            user=current_user(db)
            if user is None:return jsonify(error='Login required'),401
            rows=db.scalars(select(Course).join(Favorite,Favorite.course_id==Course.id)
                            .where(Favorite.user_id==user.id).order_by(Course.id)).all()
            return jsonify(items=[dict(id=c.id,title=c.title,category=c.category) for c in rows])
    return app

if __name__=='__main__':
    app=create_app(os.environ['DATABASE_URL'],os.environ['SECRET_KEY'],
                   secure_cookies=os.environ.get('HTTPS_COOKIES')=='1')
    app.run(host='127.0.0.1',port=5000,debug=False)
