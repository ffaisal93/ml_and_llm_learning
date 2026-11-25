import math,time,json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class CustomMNIST(MNIST):
    mirrors=['https://raw.githubusercontent.com/mkolod/MNIST/master/']
    resources=[('train-images-idx3-ubyte.gz','f68b3c2dcbeaaa9fbdd348bbdeb94873'),('train-labels-idx1-ubyte.gz','d53e105ee54ea40749a09fcbcd1e9432'),('t10k-images-idx3-ubyte.gz','9fb629c4189551a2d022fa330f9573f3'),('t10k-labels-idx1-ubyte.gz','ec29112dd5afa0611ce80d1b7f02629c')]

class MNISTSequenceDataset(Dataset):
    def __init__(self,train=True,data_dir='../data',samples_per_class=100):
        self.dataset=CustomMNIST(root=data_dir,train=train,download=True,transform=transforms.Compose([transforms.Resize((7,7)),transforms.ToTensor()]))
        self.indices=[]
        if train:
            for c in range(10):self.indices.extend((self.dataset.targets==c).nonzero(as_tuple=False).view(-1)[:samples_per_class].tolist())
        else:self.indices=list(range(len(self.dataset)))
    def __len__(self):return len(self.indices)
    def __getitem__(self,idx):
        image,_=self.dataset[self.indices[idx]]
        return (image.view(-1)*255).long()

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads,dropout):
        super().__init__();assert embedding_dim%num_heads==0
        self.num_heads=num_heads;self.head_dim=embedding_dim//num_heads
        self.q_proj=nn.Linear(embedding_dim,embedding_dim);self.k_proj=nn.Linear(embedding_dim,embedding_dim)
        self.v_proj=nn.Linear(embedding_dim,embedding_dim);self.out_proj=nn.Linear(embedding_dim,embedding_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,past=None,use_cache=False):
        B,T,C=x.shape;H=self.num_heads;D=self.head_dim
        q,k,v=[proj(x).view(B,T,H,D).transpose(1,2) for proj in (self.q_proj,self.k_proj,self.v_proj)]
        offset=0 if past is None else past[0].shape[2]
        if past is not None:k,v=torch.cat([past[0],k],2),torch.cat([past[1],v],2)
        scores=q@k.transpose(-2,-1)/math.sqrt(D)
        future=torch.arange(k.shape[2],device=x.device)[None,:]>(offset+torch.arange(T,device=x.device))[:,None]
        probs=self.dropout(scores.masked_fill(future,float('-inf')).softmax(-1))
        y=(probs@v).transpose(1,2).contiguous().view(B,T,C)
        y=self.out_proj(y)
        return (y,(k,v)) if use_cache else y

class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,dropout):
        super().__init__();self.attn=MaskedMultiHeadSelfAttention(embedding_dim,num_heads,dropout)
        self.ln1=nn.LayerNorm(embedding_dim);self.ln2=nn.LayerNorm(embedding_dim)
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,4*embedding_dim),nn.ReLU(),nn.Linear(4*embedding_dim,embedding_dim))
    def forward(self,x,past=None,use_cache=False):
        result=self.attn(self.ln1(x),past,use_cache);a,cache=result if use_cache else (result,None)
        x=x+a;x=x+self.mlp(self.ln2(x))
        return (x,cache) if use_cache else x

class SimpleGPT(nn.Module):
    def __init__(self,vocab_size=256,max_seq_len=49,embedding_dim=128,num_heads=4,num_layers=4,dropout=.1):
        super().__init__();self.token_embedding=nn.Embedding(vocab_size,embedding_dim)
        self.position_encoding=nn.Embedding(max_seq_len,embedding_dim)
        self.layers=nn.ModuleList([TransformerBlock(embedding_dim,num_heads,dropout) for _ in range(num_layers)])
        self.ln_f=nn.LayerNorm(embedding_dim);self.head=nn.Linear(embedding_dim,vocab_size)
    def forward(self,tokens,past=None,use_cache=False):
        B,T=tokens.shape
        if past is not None and len(past)!=len(self.layers):raise ValueError('one cache per layer required')
        offset=0 if past is None else past[0][0].shape[2]
        if offset+T>self.position_encoding.num_embeddings:raise ValueError('context overflow')
        x=self.token_embedding(tokens)+self.position_encoding(torch.arange(offset,offset+T,device=tokens.device))
        caches=[]
        for i,layer in enumerate(self.layers):
            result=layer(x,None if past is None else past[i],use_cache);x,cache=result if use_cache else (result,None)
            if use_cache:caches.append(cache)
        logits=self.head(self.ln_f(x))
        return (logits,caches) if use_cache else logits

def train(model,dataloader,optimizer,criterion,device):
    model.train();total_loss=0.
    for batch in dataloader:
        batch=batch.to(device);optimizer.zero_grad();logits=model(batch[:,:-1])
        loss=criterion(logits.reshape(-1,logits.size(-1)),batch[:,1:].reshape(-1))
        loss.backward();optimizer.step();total_loss+=loss.item()
    return total_loss/len(dataloader)

@torch.no_grad()
def generate_batch_no_cache(model,start_tokens,seq_len,device):
    model.eval();tokens=start_tokens[:,None].to(device)
    for _ in range(seq_len-1):
        probs=model(tokens)[:,-1,:].softmax(-1)
        tokens=torch.cat([tokens,torch.multinomial(probs,1)],1)
    return tokens

@torch.no_grad()
def generate_batch_with_cache(model,start_tokens,seq_len,device):
    model.eval();tokens=start_tokens[:,None].to(device);past=None;current=tokens
    for _ in range(seq_len-1):
        logits,past=model(current,past,use_cache=True);current=torch.multinomial(logits[:,-1,:].softmax(-1),1)
        tokens=torch.cat([tokens,current],1)
    return tokens

@torch.no_grad()
def check(model,device):
    model.eval();tokens=torch.arange(12,device=device)[None,:].repeat(2,1)
    changed=tokens.clone();changed[:,7:]+=40
    assert torch.allclose(model(tokens)[:,:7],model(changed)[:,:7],atol=1e-6)
    past=None;worst=0.
    for t in range(tokens.shape[1]):
        cached,past=model(tokens[:,t:t+1],past,use_cache=True);full=model(tokens[:,:t+1])[:,-1:]
        error=(cached-full).abs().max().item();worst=max(worst,error);assert error<1e-4,error
    start=torch.zeros(100,dtype=torch.long)
    torch.manual_seed(123);a=generate_batch_no_cache(model,start,49,device)
    torch.manual_seed(123);b=generate_batch_with_cache(model,start,49,device)
    assert a.shape==b.shape==(100,49) and torch.all(a[:,0]==0)
    timings=[]
    for fn in (generate_batch_no_cache,generate_batch_with_cache):
        values=[]
        for _ in range(3):
            if device.type=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter();fn(model,start,49,device)
            if device.type=='cuda':torch.cuda.synchronize()
            values.append(time.perf_counter()-tick)
        timings.append(sorted(values)[1])
    return dict(cache_logit_max_error=worst,same_seed_samples_equal=torch.equal(a,b),no_cache_seconds=timings[0],cache_seconds=timings[1],speedup=timings[0]/timings[1])

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',default='../data');parser.add_argument('--out',default='w005_output');args=parser.parse_args()
    torch.set_num_threads(4)  # CPU scheduling, not a changed learning hyperparameter.
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=MNISTSequenceDataset(data_dir=args.data_dir);loader=DataLoader(ds,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0);model=SimpleGPT(256,49,32,4,4,.1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-2);criterion=nn.CrossEntropyLoss();losses=[]
    for epoch in range(1,31):
        loss=train(model,loader,optimizer,criterion,device);losses.append(loss);print(f'Epoch {epoch}, Loss: {loss:.4f}',flush=True)
    stats=check(model,device);stats['last_epoch_loss']=losses[-1]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'w005_metrics.json').write_text(json.dumps(stats,indent=2));torch.save(model.state_dict(),out/'w005_model.pt')
    torch.manual_seed(123);images=generate_batch_no_cache(model,torch.zeros(16,dtype=torch.long),49,device)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(6,6))
    for ax,image in zip(axes.flat,images.cpu()):ax.imshow(image.reshape(7,7),cmap='gray',vmin=0,vmax=255);ax.axis('off')
    fig.savefig(out/'w005_samples.png');plt.close(fig);print(json.dumps(stats),flush=True)
    assert losses[-1]<2.8,'exact-data loss target not reached'
