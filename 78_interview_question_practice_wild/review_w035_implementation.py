import numpy as np

def bounds(length,rank,world_size):
    q,r=divmod(length,world_size)
    start=rank*q+min(rank,r)
    return start,start+q+(rank<r)

def dp_mat_mul(A,B,rank,world_size,comm):
    start,end=bounds(A.shape[0],rank,world_size)
    local_C=A[start:end]@B
    return np.concatenate(comm.all_gather(rank,local_C),axis=0)

def fsdp_mat_mul(A,B,rank,world_size,comm):
    lo,hi=bounds(B.shape[1],rank,world_size)
    local_B=B[:,lo:hi].copy()
    # B is a full input only in this simulator's function signature; the
    # distributed computation uses the local shard and the gathered parameters.
    full_B=np.concatenate(comm.all_gather(rank,local_B),axis=1)
    start,end=bounds(A.shape[0],rank,world_size)
    local_C=A[start:end]@full_B
    return np.concatenate(comm.all_gather(rank,local_C),axis=0)

# Chosen standalone communication fixture; use the provided comm in the exercise.
class Communicator:
    def __init__(self,world_size):
        import threading
        self.world_size=world_size;self.barrier=threading.Barrier(world_size,timeout=10)
        self.slots=[None]*world_size;self.calls=[0]*world_size
    def all_gather(self,rank,data):
        from copy import deepcopy
        self.calls[rank]+=1;self.slots[rank]=deepcopy(data);self.barrier.wait()
        result=deepcopy(self.slots);self.barrier.wait()
        if rank==0:self.slots=[None]*self.world_size
        self.barrier.wait();return result

def run_all(fn,A,B,world_size):
    from concurrent.futures import ThreadPoolExecutor
    comm=Communicator(world_size)
    def worker(rank):
        try:return fn(A,B,rank,world_size,comm)
        except BaseException:
            comm.barrier.abort();raise
    with ThreadPoolExecutor(max_workers=world_size) as pool:
        results=list(pool.map(worker,range(world_size)))
    return results,comm.calls

if __name__=='__main__':
    rng=np.random.default_rng(3)
    for m,k,n,p in [(4,3,5,2),(8,8,8,4),(7,5,6,3),(2,3,1,5),(3,2,4,1)]:
        A=rng.normal(size=(m,k));B=rng.normal(size=(k,n))
        for fn,count in [(dp_mat_mul,1),(fsdp_mat_mul,2)]:
            outputs,calls=run_all(fn,A,B,p)
            assert all(np.allclose(C,A@B) for C in outputs) and calls==[count]*p
    print('W035: every rank matches A@B; rectangular/uneven/empty shards and collective counts passed')
