"""W074: injected tokenizer/model and reproducible empirical bootstrap."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class Sampler:
    def __init__(self,encode,decode,next_logits,eos_id):
        self.encode=encode; self.decode=decode
        self.next_logits=next_logits; self.eos_id=eos_id
    def tokenize(self,text): return list(self.encode(text))
    def detokenize(self,tokens): return self.decode(tokens)
    def sample(self,prompt,max_new_tokens=32,temperature=1.,seed=0):
        if max_new_tokens<0 or temperature<=0:
            raise ValueError('Need nonnegative limit and positive temperature')
        rng=np.random.default_rng(seed)
        prefix=self.tokenize(prompt); generated=[]
        for _ in range(max_new_tokens):
            logits=np.asarray(self.next_logits(prefix+generated),dtype=float)
            if logits.ndim!=1 or not len(logits) or not np.isfinite(logits).all():
                raise ValueError('Expected a finite vocabulary logit vector')
            scaled=(logits-logits.max())/temperature
            probabilities=np.exp(scaled); probabilities/=probabilities.sum()
            token=int(rng.choice(len(logits),p=probabilities))
            if token==self.eos_id: break
            generated.append(token)
        return self.detokenize(generated)

def reward_interval(samples,reward,replicates=10000,seed=0):
    if not samples or replicates<1:
        raise ValueError('Need samples and positive replicate count')
    values=np.asarray([reward(sample) for sample in samples],dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Rewards must be finite scalar values')
    rng=np.random.default_rng(seed)
    means=np.empty(replicates)
    # One replicate at a time avoids an enormous replicates × samples allocation.
    for b in range(replicates):
        means[b]=rng.choice(values,size=len(values),replace=True).mean()
    low,high=np.quantile(means,[.005,.995])
    return float(values.mean()),(float(low),float(high))

def sample_many(sampler,prompts,seeds,workers=4):
    if len(prompts)!=len(seeds): raise ValueError('One seed per sample')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda pair:sampler.sample(pair[0],seed=pair[1]),zip(prompts,seeds)))
