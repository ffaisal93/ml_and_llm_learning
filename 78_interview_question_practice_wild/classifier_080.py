"""W080: runnable classifier pipeline, with an injected BERT-compatible encoder."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

class TextDataset(Dataset):
    def __init__(self,rows,tokenizer,max_length=32):
        self.rows=rows; self.tokenizer=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        text,label=self.rows[index]
        encoded=self.tokenizer(text,padding='max_length',truncation=True,
                               max_length=self.max_length,return_tensors='pt')
        item={key:value.squeeze(0) for key,value in encoded.items()}
        item['labels']=torch.tensor(int(label),dtype=torch.long)
        return item

class Classifier(nn.Module):
    def __init__(self,encoder,hidden_size,classes=2):
        super().__init__()
        self.encoder=encoder; self.dropout=nn.Dropout(.1)
        self.head=nn.Linear(hidden_size,classes)
    def forward(self,input_ids,attention_mask,**kwargs):
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask,**kwargs).last_hidden_state
        mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.head(self.dropout(pooled))

def train_epoch(model,loader,optimizer):
    model.train(); total=0.; count=0
    device=next(model.parameters()).device
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        labels=batch.pop('labels')
        optimizer.zero_grad(set_to_none=True)
        logits=model(**batch)
        loss=nn.functional.cross_entropy(logits,labels)  # raw B,C logits, B long labels
        loss.backward(); optimizer.step()
        total+=loss.item()*len(labels); count+=len(labels)
    return total/count

@torch.no_grad()
def evaluate(model,loader):
    from sklearn.metrics import roc_auc_score
    model.eval(); labels=[]; probabilities=[]
    device=next(model.parameters()).device
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        y=batch.pop('labels')
        probabilities.append(model(**batch).softmax(-1).cpu())
        labels.append(y.cpu())
    y=torch.cat(labels).numpy(); p=torch.cat(probabilities).numpy()
    if len(set(y.tolist()))!=p.shape[1]:
        raise ValueError('AUC requires both classes, or all classes for this multiclass evaluation')
    auc=roc_auc_score(y,p[:,1]) if p.shape[1]==2 else roc_auc_score(
        y,p,multi_class='ovr',average='macro',labels=list(range(p.shape[1])))
    return float(auc)

class ToyTokenizer:
    def __call__(self,text,**kwargs):
        ids=torch.zeros(1,kwargs['max_length'],dtype=torch.long)
        ids[0,0]=1 if text=='positive' else 2
        mask=torch.zeros_like(ids); mask[0,0]=1
        return dict(input_ids=ids,attention_mask=mask)

class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.embeddings=nn.Embedding(3,8)
    def forward(self,input_ids,attention_mask):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))

if __name__=='__main__':
    torch.manual_seed(3); torch.set_num_threads(1)
    rows=[('positive','1'),('negative','0')]*8
    loader=DataLoader(TextDataset(rows,ToyTokenizer()),batch_size=16)
    model=Classifier(ToyEncoder(),8)
    assert all(p.requires_grad for p in model.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    before=train_epoch(model,loader,optimizer)
    for _ in range(79): after=train_epoch(model,loader,optimizer)
    auc=evaluate(model,loader)
    assert after<.01 and auc==1.
    print('Toy-batch loss',before,'->',after,'AUC',auc)
