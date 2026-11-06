# Additional algorithms and worked arithmetic

<span class="practice-page"></span>

These references support the extended question bank. They specify a small exercise contract rather than pretending to be production kernels or complete distributed services.

Run all checks from the repository root, with NumPy available:

```bash
python3 -m unittest discover -s 78_interview_question_practice_wild -p 'test_*.py' -v
```

The new checks cover compiler liveness, idle-interval union, stopping-time mass at the forced endpoint, finite-action minimax, Manhattan k-medians, the bag/grid recurrences, dice enumeration, SPD two-cycles, the darts optimum, harvesting arithmetic, and the new SQL queries. The bank-coverage checks separately verify all 547 entries and their six groups.

## Contracts and complexity

| Helper | Contract | Cost / limitation |
|---|---|---|
| `compiler_cost` | Valid straight-line assignment syntax from the exercise | O(total tokens), storing statements and last uses |
| `GPUIdleIntervalTracker` | Sorted half-open busy union per node | O(m) list insertion/query; no cross-node state |
| `cumulative_deduped_time` | Union durations per run | O(n log n) sorting; zero-length intervals add zero |
| `stopping_distribution` | Forced stop on final step | O(N) time/output storage; final mass includes all survival |
| `minimax_decision` | Explicit finite actions/adversaries and reward callback | O(actions * adversaries * horizon) |
| `manhattan_pickups` | Lower-median alternating local solution | No global optimality guarantee; centers initialized from the first K input points |
| `expected_remaining_balls` | Uniform draws without replacement until a color empties | Memoized finite state recurrence |
| `grid_exit_time` | Unit square-lattice walk with absorbing boundary | Dense linear solve; a small numerical reference, not a large-grid solver |

The k-median history records objective before updates; the returned final objective also must not increase. A production implementation needs stronger input-size and numerical protections. The compiler assumes valid identifiers/operators and the given single-assignment ordering; it is not a general parser or evaluator.

## Executable references

```python
"""Additional small reference algorithms; Python 3.9+ and NumPy."""
from collections import defaultdict
from functools import lru_cache
import math
import re
import numpy as np


def compiler_cost(program):
    """Linear token scan under the exercise's cost/liveness model."""
    statements=[]
    for line in program.splitlines():
        if not line.strip(): continue
        lhs,rhs=line.split('=',1)
        tokens=re.findall(r'[A-Za-z_]\w*|\d+|[+*/-]',rhs)
        statements.append((lhs.strip(),tokens))
    assigned={lhs for lhs,_ in statements}
    last_use={}
    total=0
    frees=defaultdict(set)
    for index,(_,tokens) in enumerate(statements):
        total+=1
        for token in tokens:
            if token in '+-':total+=1
            elif token in '*/':total+=5
            elif not token.isdigit():
                total+=1
                if token in assigned:last_use[token]=index
    for name,index in last_use.items():frees[index].add(name)
    live=set();peak=0
    for index,(lhs,_) in enumerate(statements):
        live.difference_update(frees[index])
        live.add(lhs)
        peak=max(peak,len(live))
    return total,peak


def merge_intervals(intervals):
    result=[]
    for start,end in sorted(intervals):
        if start>=end:raise ValueError('interval must be nonempty')
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((start,end))
    return result


class GPUIdleIntervalTracker:
    def __init__(self): self.busy=defaultdict(list)
    def add_task(self,start,end,node_id):
        if start>=end:raise ValueError('invalid task')
        self.busy[node_id]=merge_intervals(self.busy[node_id]+[(start,end)])
    def get_idle_intervals(self,node_id,window_start,window_end):
        if window_start>window_end:raise ValueError('invalid window')
        cursor=window_start;out=[]
        for start,end in self.busy.get(node_id,[]):
            if end<=cursor:continue
            if start>=window_end:break
            if start>cursor:out.append((cursor,min(start,window_end)))
            cursor=max(cursor,min(end,window_end))
        if cursor<window_end:out.append((cursor,window_end))
        return out


def cumulative_deduped_time(rows):
    groups=defaultdict(list)
    for row in rows:
        start,end=row['start_time'],row['end_time']
        if start>end:raise ValueError('negative interval')
        if start<end:groups[row['run_id']].append((start,end))
        else:groups[row['run_id']]
    return {key:sum(end-start for start,end in merge_intervals(value)) for key,value in groups.items()}


def stopping_distribution(p):
    """p includes the final step, where remaining mass is forced to stop."""
    if not p or any(not 0<=x<=1 for x in p):raise ValueError('invalid probabilities')
    masses=[];survival=1.0
    for probability in p[:-1]:
        masses.append(survival*probability)
        survival*=1-probability
    masses.append(survival)
    return masses,sum((i+1)*mass for i,mass in enumerate(masses))


def minimax_decision(decisions, adversary_probabilities, reward):
    """Finite action/adversary model. Ties keep the first supplied action."""
    if not decisions or not adversary_probabilities:raise ValueError('empty choices')
    distributions=[stopping_distribution(p)[0] for p in adversary_probabilities]
    scored=[]
    for decision in decisions:
        expected=[sum(mass*reward(decision,t+1,adversary) for t,mass in enumerate(dist))
                  for adversary,dist in enumerate(distributions)]
        scored.append(min(expected))
    best=max(range(len(decisions)),key=lambda i:scored[i])
    return decisions[best],scored[best]


def manhattan_pickups(points,k,max_iter=100):
    """Simple seeded-by-input local solution; no global guarantee."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or not 1<=k<=len(points) or max_iter<1:
        raise ValueError('invalid points/k')
    if not np.isfinite(points).all():raise ValueError('nonfinite point')
    centers=points[:k].copy()
    history=[]
    for _ in range(max_iter):
        distances=np.abs(points[:,None,:]-centers[None,:,:]).sum(-1)
        labels=distances.argmin(1)
        history.append(float(distances[np.arange(len(points)),labels].sum()))
        updated=centers.copy()
        for j in range(k):
            members=points[labels==j]
            if len(members):updated[j]=np.sort(members,axis=0)[(len(members)-1)//2]
        if np.array_equal(updated,centers):break
        centers=updated
    distances=np.abs(points[:,None,:]-centers[None,:,:]).sum(-1)
    labels=distances.argmin(1)
    total=float(distances[np.arange(len(points)),labels].sum())
    return centers,labels,total,history


@lru_cache(None)
def expected_remaining_balls(a,b,c):
    if min(a,b,c)==0:return float(a+b+c)
    return (a*expected_remaining_balls(a-1,b,c)+b*expected_remaining_balls(a,b-1,c)
            +c*expected_remaining_balls(a,b,c-1))/(a+b+c)


def grid_exit_time(side):
    """Boundary coordinates 0..side, start at floor(side/2) in each axis."""
    if side<2:raise ValueError('need an interior start')
    width=side-1;n=width*width
    matrix=np.eye(n);rhs=np.ones(n)
    for x in range(1,side):
        for y in range(1,side):
            row=(x-1)*width+y-1
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                xx,yy=x+dx,y+dy
                if 1<=xx<side and 1<=yy<side:matrix[row,(xx-1)*width+yy-1]-=.25
    values=np.linalg.solve(matrix,rhs)
    start=side//2
    return float(values[(start-1)*width+start-1])

```

## Deduplicated intervals in SQL

For rows `activity(run_id,start_time,end_time)`, this SQLite-compatible query uses the maximum prior end, not only the immediately preceding interval end. That distinction handles nested overlapping intervals correctly. Validate `start_time <= end_time` before aggregation.

```sql
WITH ordered AS (
 SELECT *,MAX(end_time) OVER (PARTITION BY run_id ORDER BY start_time,end_time
   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_end
 FROM activity
), marked AS (
 SELECT *,CASE WHEN prior_end IS NULL OR start_time>prior_end THEN 1 ELSE 0 END AS new_group
 FROM ordered
), islands AS (
 SELECT *,SUM(new_group) OVER (PARTITION BY run_id ORDER BY start_time,end_time
   ROWS UNBOUNDED PRECEDING) AS island
 FROM marked
), merged AS (
 SELECT run_id,island,MIN(start_time) AS s,MAX(end_time) AS e
 FROM islands GROUP BY run_id,island
)
SELECT run_id,SUM(e-s) AS total_duration FROM merged GROUP BY run_id;
```
