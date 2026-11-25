from collections import Counter

class BPETokenizer:
    def __init__(self):self.vocab={};self.inv_vocab={};self.merges=[]
    @staticmethod
    def _merge(tokens,pair):
        out=[];i=0
        while i<len(tokens):
            if i+1<len(tokens) and (tokens[i],tokens[i+1])==pair:
                out.append(tokens[i]+tokens[i+1]);i+=2
            else:out.append(tokens[i]);i+=1
        return out
    def train(self,text,threshold):
        if threshold<1:raise ValueError('positive frequency threshold required')
        self.vocab={c:i for i,c in enumerate(sorted(set(text)))};self.merges=[]
        tokens=list(text)
        while len(tokens)>1:
            counts=Counter(zip(tokens,tokens[1:]));pair=min(counts,key=lambda p:(-counts[p],p))
            if counts[pair]<threshold:break
            merged=''.join(pair)
            if merged not in self.vocab:self.vocab[merged]=len(self.vocab)
            self.merges.append(pair);tokens=self._merge(tokens,pair)
        self.inv_vocab={i:t for t,i in self.vocab.items()}
    def encode(self,text):
        tokens=list(text)
        for pair in self.merges:tokens=self._merge(tokens,pair)
        return [self.vocab[t] for t in tokens]
    def decode(self,embedding):return ''.join(self.inv_vocab[i] for i in embedding)

t=BPETokenizer();t.train('abab',2)
assert t.vocab=={'a':0,'b':1,'ab':2} and t.encode('abab')==[2,2]
assert t.decode(t.encode('baba'))=='baba'
t.train('aaaaa',2)
assert t.merges[0]==('a','a') and t.decode(t.encode('aaaaa'))=='aaaaa'
# Equal counts ab and bc: lexical pair ab wins.
t.train('abc',1);assert t.merges==[('a','b'),('ab','c')]
assert t.encode('abc')==[4] and t.decode([4])=='abc'
t.train('',1);assert t.encode('')==[] and t.decode([])==''
print('W013: frequency threshold, exact ids, lexical ties, nonoverlap and roundtrip passed')
