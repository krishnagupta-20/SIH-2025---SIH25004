"""Fine-tune MobileNetV3 on folders: data/{train,val}/{cattle__Gir,buffalo__Murrah,...}."""
import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--data',default='data'); p.add_argument('--epochs',type=int,default=12)
 p.add_argument('--batch-size',type=int,default=24); p.add_argument('--image-size',type=int,default=224)
 p.add_argument('--output',default='model.onnx'); p.add_argument('--freeze-backbone',action='store_true')
 a=p.parse_args(); root=Path(a.data); size=a.image_size
 norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
 train_tf=transforms.Compose([transforms.Resize((size,size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ColorJitter(.12,.12,.12,.04),transforms.ToTensor(),norm])
 val_tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),norm])
 train=datasets.ImageFolder(root/'train',transform=train_tf); val=datasets.ImageFolder(root/'val',transform=val_tf)
 if train.classes!=val.classes: raise ValueError('Train and val folders must contain the same breed class folders.')
 if len(train.classes)<2: raise ValueError('Provide at least two breed classes.')
 device='cuda' if torch.cuda.is_available() else 'cpu'
 tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0); va=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
 model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
 if a.freeze_backbone:
  for q in model.features.parameters(): q.requires_grad=False
 model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(train.classes)); model.to(device)
 opt=torch.optim.AdamW((q for q in model.parameters() if q.requires_grad),lr=2e-4,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss(); best=0.
 for epoch in range(a.epochs):
  model.train(); total=correct=seen=0
  for x,y in tr:
   x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step(); total+=loss.item()*y.size(0); correct+=(logits.argmax(1)==y).sum().item(); seen+=y.size(0)
  model.eval(); hits=count=0
  with torch.no_grad():
   for x,y in va:
    pred=model(x.to(device)).argmax(1).cpu(); hits+=(pred==y).sum().item(); count+=len(y)
  accuracy=hits/max(count,1); print(f'Epoch {epoch+1}/{a.epochs}: train loss {total/max(seen,1):.4f}, train accuracy {correct/max(seen,1):.3f}, validation accuracy {accuracy:.3f}')
  if accuracy>best: best=accuracy; torch.save(model.cpu().state_dict(),'best_weights.pt'); model.to(device)
 if not Path('best_weights.pt').exists(): raise RuntimeError('Training did not produce a model checkpoint.')
 model.load_state_dict(torch.load('best_weights.pt',map_location='cpu',weights_only=True)); model.eval()
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 torch.onnx.export(model,torch.zeros(1,3,size,size),str(out),input_names=['images'],output_names=['logits'],dynamic_axes={'images':{0:'batch'},'logits':{0:'batch'}},opset_version=17)
 labels=[]
 for c in train.classes:
  if '__' not in c: raise ValueError(f'Class folder {c!r} must start with cattle__ or buffalo__.')
  species,name=c.split('__',1); species=species.lower()
  if species not in {'cattle','buffalo'}: raise ValueError(f'Unknown species prefix in {c!r}.')
  labels.append({'name':name.replace('_',' '),'species':species,'crossbred':'crossbred' in name.lower()})
 metadata={'input_size':[size,size],'input_layout':'NCHW','mean':[.485,.456,.406],'std':[.229,.224,.225],'output_kind':'logits','input_name':'images','output_name':'logits','validation_accuracy':best,'labels':labels}
 out.with_name('breeds.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 print(f'Exported {out} and {out.with_name("breeds.json")} (best validation accuracy {best:.3f}).')
if __name__=='__main__': main()
