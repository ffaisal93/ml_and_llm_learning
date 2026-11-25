import numpy as np


def fit_gmm(X,k,max_iters=100,tol=1e-4,eps=1e-6,seed=0):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=2 or not len(X) or not 1<=k<=len(X) or max_iters<1 or tol<0 or eps<=0:
        raise ValueError('invalid data/configuration')
    if not np.isfinite(X).all():raise ValueError('finite input required')
    n,d=X.shape;rng=np.random.default_rng(seed)
    mu=X[rng.choice(n,k,replace=False)].copy()
    # K-means initialization; ties go to the first centroid, empties stay fixed.
    for _ in range(30):
        distances=np.stack([np.sum((X-center)**2,axis=1) for center in mu],axis=1)
        labels=distances.argmin(1);updated=mu.copy()
        for j in range(k):
            if np.any(labels==j):updated[j]=X[labels==j].mean(0)
        if np.array_equal(mu,updated):break
        mu=updated
    # Diagonal covariances, returned later as full d*d diagonal matrices.
    var=np.tile(np.maximum(X.var(0),eps),(k,1))
    pi=np.full(k,1/k)
    def e_step():
        log_joint=np.empty((n,k))
        for j in range(k):
            log_joint[:,j]=np.log(pi[j])-.5*(d*np.log(2*np.pi)+np.log(var[j]).sum()+((X-mu[j])**2/var[j]).sum(1))
        maximum=log_joint.max(1,keepdims=True)
        log_norm=maximum+np.log(np.exp(log_joint-maximum).sum(1,keepdims=True))
        return np.exp(log_joint-log_norm),float(log_norm.sum())
    gamma,ll=e_step();history=[ll]
    for _ in range(max_iters):
        nk=gamma.sum(0)
        for j in range(k):
            if nk[j]<1e-12:
                # Documented recovery; this restart can lower likelihood.
                mu[j]=X[rng.integers(n)];var[j]=np.maximum(X.var(0),eps)
                nk[j]=1e-12
            else:
                mu[j]=gamma[:,j]@X/nk[j]
                var[j]=(gamma[:,j,None]*(X-mu[j])**2).sum(0)/nk[j]+eps
        pi=nk/nk.sum()
        gamma,new_ll=e_step();history.append(new_ll)
        improvement=new_ll-ll;ll=new_ll
        if 0<=improvement<tol:break
    Sigma=np.array([np.diag(v) for v in var])
    return pi,mu,Sigma,gamma,history

X=np.array([[0.],[.1],[10.],[10.1]])
pi,mu,Sigma,gamma,history=fit_gmm(X,2,max_iters=50)
assert np.allclose(np.sort(mu[:,0]),[.05,10.05],atol=.01)
assert np.isclose(pi.sum(),1) and np.allclose(gamma.sum(1),1)
assert np.all(Sigma[:,0,0]>0) and np.isfinite(history).all()
assert np.min(np.diff(history))>-1e-5
pi,mu,Sigma,gamma,_=fit_gmm([[2.,2.]],1)
assert np.allclose(mu,[[2,2]]) and np.isfinite(Sigma).all() and gamma.tolist()==[[1.]]
print('W130: separated means, normalized responsibilities, regularization and singleton passed')
