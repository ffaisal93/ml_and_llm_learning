import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel,BertConfig


class KeyphraseExtractor(nn.Module):
    def __init__(self,bert_model_name='bert-base-uncased',config=None):
        super().__init__()
        self.bert=BertModel(config) if config is not None else BertModel.from_pretrained(bert_model_name)
        self.linear=nn.Linear(self.bert.config.hidden_size,2)

    def forward(self,input_ids,attention_mask,span_mask=None):
        hidden=self.bert(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        logits=self.linear(hidden)  # (B,L,2)
        valid=attention_mask.bool() if span_mask is None else attention_mask.bool() & span_mask.bool()
        if not valid.any(dim=1).all():raise ValueError('each passage needs an eligible token')
        logits=logits.masked_fill(~valid[:,:,None],float('-inf'))
        return logits[:,:,0],logits[:,:,1]

    def loss(self,batch):
        start,end=self(batch['input_ids'],batch['attention_mask'],batch.get('span_mask'))
        s,e=batch['start_idx'],batch['end_idx']
        if ((s<0)|(e<s)|(e>=start.shape[1])).any():raise ValueError('invalid span indices')
        row=torch.arange(len(s),device=s.device)
        if not (torch.isfinite(start[row,s]) & torch.isfinite(end[row,e])).all():
            raise ValueError('gold span hits a masked token')
        return (F.cross_entropy(start,s)+F.cross_entropy(end,e))/2

    @torch.no_grad()
    def predict(self,input_ids,attention_mask,span_mask=None,max_span=32):
        self.eval();start,end=self(input_ids,attention_mask,span_mask)
        length=start.shape[1];pos=torch.arange(length,device=start.device)
        legal=(pos[None,:]>=pos[:,None]) & ((pos[None,:]-pos[:,None])<max_span)
        scores=(start[:,:,None]+end[:,None,:]).masked_fill(~legal[None,:,:],float('-inf'))
        best=scores.flatten(1).argmax(1)
        return torch.stack([best//length,best%length],dim=1)


def train(model,batches,epochs=3,lr=2e-5):
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    history=[]
    for _ in range(epochs):
        model.train();total=count=0
        for batch in batches:
            optimizer.zero_grad();loss=model.loss(batch);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total+=loss.item()*len(batch['input_ids']);count+=len(batch['input_ids'])
        if not count:raise ValueError('empty/reused exhausted training iterator')
        history.append(total/count)
    return history

if __name__=='__main__':
    torch.manual_seed(0);torch.set_num_threads(1)
    # Offline smoke fixture: real BertModel, tiny randomly initialized config.
    # ids 1/2 are designated special tokens here, 0 padding, 9 the key word.
    config=BertConfig(vocab_size=32,hidden_size=24,num_hidden_layers=1,num_attention_heads=4,
                      intermediate_size=48,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    model=KeyphraseExtractor(config=config)
    ids=torch.tensor([[1,4,9,5,2,0],[1,9,6,7,2,0],[1,6,7,9,2,0],[1,8,9,6,2,0]])
    mask=ids!=0;span_mask=mask & (ids!=1) & (ids!=2)
    target=(ids==9).long().argmax(1)
    batch={'input_ids':ids,'attention_mask':mask,'span_mask':span_mask,'start_idx':target,'end_idx':target}
    losses=train(model,[batch],epochs=80,lr=.01)
    predicted=model.predict(ids,mask,span_mask)
    assert torch.equal(predicted[:,0],target) and torch.equal(predicted[:,1],target)
    assert losses[-1]<losses[0]*.1
    # Special/padded positions can never win constrained decoding.
    assert (predicted[:,1]>=predicted[:,0]).all()
    print(f'W148: tiny BERT fixture loss {losses[0]:.4f}->{losses[-1]:.4f}; spans {predicted.tolist()}')
