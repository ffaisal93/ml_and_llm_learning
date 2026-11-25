from fractions import Fraction
from itertools import combinations


def minimax_radius(location,k,integer_radius=False):
    if not location or k<1:raise ValueError('nonempty locations and positive k required')
    values=sorted(Fraction(v) for v in location)
    if k>=len(values):return 0 if integer_radius else Fraction(0)
    def feasible(diameter):
        used=0;i=0
        while i<len(values):
            used+=1
            if used>k:return False
            right=values[i]+diameter
            while i<len(values) and values[i]<=right:i+=1
        return True
    if integer_radius:
        lo,hi=0,int((values[-1]-values[0]+1)//2)+1
        while lo<hi:
            mid=(lo+hi)//2
            if feasible(2*mid):hi=mid
            else:lo=mid+1
        return lo
    # Exact solution for arbitrary finite rational inputs; O(n^2) candidates.
    diameters=sorted({b-a for a in values for b in values if b>=a})
    lo,hi=0,len(diameters)-1
    while lo<hi:
        mid=(lo+hi)//2
        if feasible(diameters[mid]):hi=mid
        else:lo=mid+1
    return diameters[lo]/2

assert minimax_radius([4,1,6,7,2],2)==Fraction(3,2)
assert minimax_radius([4,1,6,7,2],2,integer_radius=True)==2
assert minimax_radius([0,10],1)==5
assert minimax_radius([2,2,2],1)==0
# Independent exhaustive contiguous-partition oracle for small integer cases.
for values in ([0,1,4,8],[0,2,3,9,10],[1,1,2,7]):
    for k in range(1,len(values)+1):
        expected=None
        for cuts in combinations(range(1,len(values)),k-1):
            bounds=(0,)+cuts+(len(values),)
            radius=max(Fraction(values[bounds[j+1]-1]-values[bounds[j]],2) for j in range(k))
            expected=radius if expected is None else min(expected,radius)
        assert minimax_radius(values,k)==expected
print('W133: real versus integer radius and exhaustive partition oracle passed')
