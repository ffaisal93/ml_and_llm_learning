BANNED_WORDS={'scam','fake','fraud'}
MAX_STRIKES=3


def find_banned_words(text):
    found=[]
    for word in text.lower().split():
        word=word.strip('.,!?')
        if word in BANNED_WORDS and word not in found:found.append(word)
    return found


class ReviewService:
    def __init__(self,repo):self.repo=repo

    def add_review(self,user_id,product_id,text):
        user=self.repo.get_user(user_id)
        if user.flagged:return 403,{'error':'user is blocked'}
        words=find_banned_words(text)
        if words:
            self._record_violation(user,words)
            return 403,{'error':'review contains banned content','words':words}
        review=self.repo.create_review(user_id,product_id,text)
        return 201,{'id':review.id}

    def edit_review(self,user_id,review_id,text):
        user=self.repo.get_user(user_id)
        if user.flagged:return 403,{'error':'user is blocked'}
        words=find_banned_words(text)
        if words:
            self._record_violation(user,words)
            return 403,{'error':'review contains banned content','words':words}
        review=self.repo.get_review(review_id)
        review.text=text
        self.repo.save_review(review)
        return 200,{'id':review.id}

    def _record_violation(self,user,words):
        user.violations+=1
        user.violated_words.extend(words)
        if user.violations>=MAX_STRIKES:user.flagged=True
        self.repo.save_user(user)


def check():
    # A chosen minimal repository fixture honoring the explicitly stated copy contract.
    from dataclasses import dataclass,field
    from copy import deepcopy
    @dataclass
    class User:
        violations:int=0
        violated_words:list=field(default_factory=list)
        flagged:bool=False
    @dataclass
    class Review:
        id:int
        text:str
    class Repo:
        def __init__(self):self.users={1:User()};self.reviews={}
        def get_user(self,i):return deepcopy(self.users[i])
        def save_user(self,u):self.users[1]=deepcopy(u)
        def get_review(self,i):return deepcopy(self.reviews[i])
        def save_review(self,r):self.reviews[r.id]=deepcopy(r)
        def create_review(self,uid,pid,text):
            r=Review(len(self.reviews)+1,text);self.save_review(r);return r
    repo=Repo();service=ReviewService(repo)
    assert service.add_review(1,7,'great')==(201,{'id':1})
    assert service.edit_review(1,1,'great, works well')==(200,{'id':1})
    assert repo.get_review(1).text=='great, works well'
    for text in ['SCAM, scam fake!','fraud','fake']:
        assert service.edit_review(1,1,text)[0]==403
        assert repo.get_review(1).text=='great, works well'
    user=repo.get_user(1)
    assert user.violations==3 and user.flagged
    assert user.violated_words==['scam','fake','fraud','fake']
    assert service.add_review(1,7,'fine')==(403,{'error':'user is blocked'})
    assert service.edit_review(1,1,'fine')==(403,{'error':'user is blocked'})
    assert repo.get_user(1).violations==3 and len(repo.reviews)==1
    repo2=Repo();s2=ReviewService(repo2)
    assert s2.add_review(1,7,'fake!')==(403,{'error':'review contains banned content','words':['fake']})
    assert not repo2.reviews and repo2.get_user(1).violations==1
    print('W137: copy-returning fixture checks passed; original repository tests not supplied')

if __name__=='__main__':check()
