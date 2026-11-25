import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r'[a-z0-9]+',text.lower())


def build_vocab(docs):
    vocab={}
    for doc in docs:
        for token in tokenize(doc):
            if token not in vocab:
                vocab[token]=len(vocab)
    return vocab


def vectorize(text,vocab):
    return dict(Counter(vocab[t] for t in tokenize(text) if t in vocab))


def cosine(u,v):
    normu=math.sqrt(sum(x*x for x in u.values()))
    normv=math.sqrt(sum(x*x for x in v.values()))
    if not normu or not normv:return 0.0
    if len(u)>len(v):u,v=v,u
    return sum(value*v.get(key,0) for key,value in u.items())/(normu*normv)


def search(query,corpus,k):
    if k<0:raise ValueError('k must be nonnegative')
    if k==0:return []
    vocab=build_vocab(corpus); q=vectorize(query,vocab)
    scores=[(i,round(cosine(q,vectorize(doc,vocab)),9)) for i,doc in enumerate(corpus)]
    return sorted(scores,key=lambda pair:(-pair[1],pair[0]))[:k]

assert tokenize('Hello, World! Hello-2024')==['hello','world','hello','2024']
assert build_vocab(['b a','a c'])=={'b':0,'a':1,'c':2}
assert vectorize('a a c z',{'b':0,'a':1,'c':2})=={1:2,2:1}
assert search('cat',['cat sat','dog','cat cat dog'],2)==[(2,.894427191),(0,.707106781)]
assert search('a',['a b','b','a b'],3)==[(0,.707106781),(2,.707106781),(1,0.)]
assert search('unknown',['a','b'],5)==[(0,0.),(1,0.)]
assert search('',[],1)==[] and search('a',['a'],0)==[]
assert search('cat OOV',['cat sat'],1)==search('cat',['cat sat'],1)
print('W129: tokenization, sparse counts, exact ranking/ties/OOV checks passed')
