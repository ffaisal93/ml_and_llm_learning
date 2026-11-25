import numpy as np

def multi_head_attention(Q,K,V,num_heads,mask=None):
    Q,K,V=[np.asarray(a,dtype=np.float64) for a in (Q,K,V)]
    if Q.ndim!=3 or Q.shape!=K.shape or Q.shape!=V.shape:raise ValueError('matching B,T,D inputs required')
    B,T,D=Q.shape
    if num_heads<1 or D%num_heads:raise ValueError('heads must divide model width')
    d=D//num_heads
    q,k,v=[a.reshape(B,T,num_heads,d).transpose(0,2,1,3) for a in (Q,K,V)]
    scores=q@k.swapaxes(-1,-2)/np.sqrt(d)
    if mask is not None:
        allowed=np.broadcast_to(np.asarray(mask,dtype=bool),scores.shape)
        if not allowed.any(axis=-1).all():raise ValueError('all-masked query')
        scores=np.where(allowed,scores,-np.inf)
    probabilities=np.exp(scores-scores.max(axis=-1,keepdims=True))
    probabilities/=probabilities.sum(axis=-1,keepdims=True)
    return (probabilities@v).transpose(0,2,1,3).reshape(B,T,D)

q=np.zeros((1,2,2));v=np.array([[[1,2],[3,4]]])
assert np.allclose(multi_head_attention(q,q,v,1),[[[2,3],[2,3]]])
assert np.allclose(multi_head_attention(q,q,v,1,[[1,0],[1,1]]),[[[1,2],[2,3]]])
rng=np.random.default_rng(3);q,k,v=[rng.normal(size=(2,3,8)) for _ in range(3)]
# Per-head oracle independently slices consecutive feature ranges.
y=multi_head_attention(q,k,v,4)
for b in range(2):
 for h in range(4):
  sl=slice(h*2,(h+1)*2);scores=q[b,:,sl]@k[b,:,sl].T/np.sqrt(2)
  p=np.exp(scores-scores.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
  assert np.allclose(y[b,:,sl],p@v[b,:,sl])
mask=np.tril(np.ones((3,3),dtype=bool));a=multi_head_attention(q,k,v,4,mask)
k[:,2]+=100;v[:,2]+=100
assert np.allclose(a[:,:2],multi_head_attention(q,k,v,4,mask)[:,:2])
assert np.isfinite(multi_head_attention(q*1000,k*1000,v,4)).all()
print('W003: examples, independent head oracle, broadcast causal mask and stability passed')
