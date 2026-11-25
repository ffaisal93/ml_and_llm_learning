"""W517 local model; API state never exposes unrevealed mines."""
from collections import deque
import random

class Minesweeper:
    def __init__(self,rows,cols,mines,seed=0):
        if rows<1 or cols<1 or not 0<=mines<rows*cols:
            raise ValueError('Need a nonempty board with at least one safe cell')
        self.rows=rows; self.cols=cols
        self.mines=set(random.Random(seed).sample(range(rows*cols),mines))
        self.revealed=set(); self.flagged=set(); self.status='playing'; self.version=0
    @classmethod
    def newGame(cls,rows,cols,mines,seed=0):return cls(rows,cols,mines,seed)
    def _index(self,r,c):
        if not 0<=r<self.rows or not 0<=c<self.cols:raise ValueError('Out of bounds')
        return r*self.cols+c
    def _neighbors(self,index):
        r,c=divmod(index,self.cols)
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if (dr or dc) and 0<=r+dr<self.rows and 0<=c+dc<self.cols:
                    yield (r+dr)*self.cols+c+dc
    def _count(self,index):return sum(n in self.mines for n in self._neighbors(index))
    def flag(self,r,c):
        index=self._index(r,c)
        if self.status!='playing' or index in self.revealed:return self.getState()
        if index in self.flagged:self.flagged.remove(index)
        else:self.flagged.add(index)
        self.version+=1
        return self.getState()
    def reveal(self,r,c):
        index=self._index(r,c)
        if self.status!='playing' or index in self.flagged or index in self.revealed:
            return self.getState()
        if index in self.mines:
            self.revealed.add(index);self.status='lost'
        else:
            queue=deque([index])
            while queue:
                current=queue.popleft()
                if current in self.revealed or current in self.flagged or current in self.mines:continue
                self.revealed.add(current)
                if self._count(current)==0:queue.extend(self._neighbors(current))
            if len(self.revealed)==self.rows*self.cols-len(self.mines):self.status='won'
        self.version+=1
        return self.getState()
    def getState(self):
        board=[]
        for r in range(self.rows):
            row=[]
            for c in range(self.cols):
                index=r*self.cols+c
                row.append('*' if index in self.revealed and index in self.mines else
                    self._count(index) if index in self.revealed else
                    'F' if index in self.flagged else '.')
            board.append(row)
        return dict(board=board,status=self.status,version=self.version)
    def snapshot(self):
        # Private persistence representation; never return this to a player.
        return dict(rows=self.rows,cols=self.cols,mines=sorted(self.mines),
                    revealed=sorted(self.revealed),flagged=sorted(self.flagged),
                    status=self.status,version=self.version)
    @classmethod
    def restore(cls,snapshot):
        game=cls(snapshot['rows'],snapshot['cols'],0)
        game.mines=set(snapshot['mines']);game.revealed=set(snapshot['revealed'])
        game.flagged=set(snapshot['flagged']);game.status=snapshot['status'];game.version=snapshot['version']
        return game

if __name__=='__main__':
    empty=Minesweeper(3,3,0);assert empty.reveal(1,1)['status']=='won'
    game=Minesweeper(2,2,1);mine=next(iter(game.mines));r,c=divmod(mine,2)
    game.flag(r,c);before=game.version;game.reveal(r,c);assert game.version==before
    game.flag(r,c);assert game.reveal(r,c)['status']=='lost'
    assert Minesweeper.restore(game.snapshot()).getState()==game.getState()
    safe=Minesweeper(3,3,1)
    for cell in set(range(9))-safe.mines:safe.reveal(*divmod(cell,3))
    assert safe.status=='won'
    assert '*' not in str(Minesweeper(2,2,1).getState())
    print('W517 flood-fill, flags, loss, win, snapshot and hidden-state checks passed')
