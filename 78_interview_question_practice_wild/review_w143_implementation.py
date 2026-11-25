import numpy as np


class SimpleTokenizer:
    def __init__(self):
        self.vocab={'<unk>':0};self.inv_vocab={0:'<unk>'}

    def tokenize(self,text):
        if isinstance(text,str):return text.split()
        return [self.tokenize(item) for item in text]

    def build_vocab(self,texts):
        self.vocab={'<unk>':0}
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:self.vocab[token]=len(self.vocab)
        self.inv_vocab={i:t for t,i in self.vocab.items()}
        return self.vocab

    def word_to_token(self,text):
        if isinstance(text,str):return [self.vocab.get(t,0) for t in self.tokenize(text)]
        return [self.word_to_token(t) for t in text]

    def token_to_word(self,ids):
        # [] is the empty single sequence; [[],[]] is a batch of two empties.
        if ids and isinstance(ids[0],(list,tuple)):
            return [self.token_to_word(row) for row in ids]
        return ' '.join(self.inv_vocab.get(int(i),'<unk>') for i in ids)


def train_bigram(token_ids,vocab_size,epochs=200,lr=.5,seed=0):
    ids=np.asarray(token_ids)
    if ids.ndim!=1 or len(ids)<2 or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('at least two integer ids required')
    if vocab_size<1 or np.any(ids<0) or np.any(ids>=vocab_size) or epochs<1 or lr<=0:
        raise ValueError('invalid training configuration')
    rng=np.random.default_rng(seed)
    W=rng.normal(0,.01,(vocab_size,vocab_size));losses=[]
    x,y=ids[:-1],ids[1:];n=len(x)
    for _ in range(epochs):
        logits=W[x];shifted=logits-logits.max(1,keepdims=True)
        logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True))
        losses.append(float(-logp[np.arange(n),y].mean()))
        g=np.exp(logp);g[np.arange(n),y]-=1;g/=n
        grad=np.zeros_like(W)
        np.add.at(grad,x,g)  # repeated current-token ids must accumulate!
        W-=lr*grad
    return W,losses

tok=SimpleTokenizer()
assert tok.tokenize('the  cat   sat')==['the','cat','sat']
assert tok.tokenize(['a b','c'])==[['a','b'],['c']]
assert tok.build_vocab(['the cat sat on the mat','the dog sat on the log'])=={'<unk>':0,'the':1,'cat':2,'sat':3,'on':4,'mat':5,'dog':6,'log':7}
assert tok.word_to_token('the cat elephant')==[1,2,0]
assert tok.token_to_word([1,2,3])=='the cat sat'
assert tok.token_to_word([[],[999]])==['','<unk>']
assert tok.token_to_word(tok.word_to_token(' the  cat\nsat '))=='the cat sat'
ids=tok.word_to_token('the cat sat on the mat the dog sat on the log')
W,losses=train_bigram(ids,8)
assert losses[-1]<losses[0] and np.isfinite(W).all()
# Independently verify a single SGD update with repeated x=1 and conflicting targets.
W0=np.random.default_rng(3).normal(0,.01,(3,3));p=np.exp(W0[1]-max(W0[1]));p/=p.sum()
expected=W0.copy();expected[1]-=.5*(p-np.array([0,.5,.5]))
W1,_=train_bigram([1,1,2],3,epochs=1,seed=3)
assert np.allclose(W1,expected)
print(f'W143: roundtrips and exact SGD update passed; losses {losses[0]:.6f} -> {losses[-1]:.6f}')
