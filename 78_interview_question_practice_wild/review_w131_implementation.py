def kmeans(points,centroids,max_iters=100):
    if not points or not centroids or max_iters<1:raise ValueError('invalid input')
    d=len(points[0]); k=len(centroids)
    if not d or k>len(points) or any(len(p)!=d for p in points+centroids):
        raise ValueError('incompatible dimensions')
    centers=[list(map(float,c)) for c in centroids]
    previous=None
    for _ in range(max_iters):
        labels=[min(range(k),key=lambda j:sum((a-b)**2 for a,b in zip(p,centers[j]))) for p in points]
        if labels==previous:break
        sums=[[0.]*d for _ in range(k)];counts=[0]*k
        for p,label in zip(points,labels):
            counts[label]+=1
            for j,value in enumerate(p):sums[label][j]+=value
        for j in range(k):
            if counts[j]:centers[j]=[v/counts[j] for v in sums[j]]
        previous=labels
    return centers,labels

points=[(0,0),(0,1),(10,10),(10,11)]; initial=[(0,0),(10,10)]
assert kmeans(points,initial)==([[0.,.5],[10.,10.5]],[0,0,1,1])
assert initial==[(0,0),(10,10)] and points[0]==(0,0)
assert kmeans([(0,),(1,),(2,),(3,),(10,)],[(0,),(1,)],1)==([[0.],[4.]],[0,1,1,1,1])
assert kmeans([(0,),(4,)],[(2,),(2,)])==([[2.],[2.]],[0,0])
print('W131: supplied centers, one-iteration labels, ties, empty cluster and nonmutation passed')
