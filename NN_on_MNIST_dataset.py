import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

#data
train_ds=dataset.MNIST(root='/root', train=True,transform=transforms.ToTensor(),download=True)
test_ds=dataset.MNIST(root='/root', train=False,transform=transforms.ToTensor(),download=True)

train_ds.data.shape 
test_ds.data.shape

batch_size=32
#dataloader  
train_dl=DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=2)
test_dl=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=True,num_workers=2)

image,target=next(iter(train_dl))
image.shape  
target   

image,target=next(iter(train_dl))
plt.figure(figsize=(5,5))
for i in range(18):
  plt.subplot(3,6,i+1)  
  plt.imshow(image[i,0])
  plt.axis('off')
plt.show() 


# define modelll
#define class
class SimpleNN(nn.Module):
   def __init__(self,num_feature,num_class):  
       super(SimpleNN,self).__init__()
       
       self.fc1=nn.Linear(in_features=num_feature,out_features=100) 
       self.fc2=nn.Linear(in_features=100,out_features=num_class) 

   def forward(self,x):    
       out=self.fc1(x)  
       out=F.relu(out)  
       out=self.fc2(out)
       return out
#end define model

device='cuda'  if torch.cuda.is_available() else 'cpu'
model=SimpleNN(28*28,10).to(device)
model


#define loss funtion 
citeration=nn.CrossEntropyLoss()

#define optimizer
optimizer=optim.Adam(model.parameters(),lr=0.01) 

epoch=10
for i  in range(epoch):
  SumLoss=0
  for idx,(data,target)  in enumerate(train_dl):  
    data=data.to(device)  
    target=target.to(device)
    data=data.reshape(data.shape[0],-1)  
    
    optimizer.zero_grad()
    score=model(data) 
    loss=citeration(score,target)  
    
    SumLoss+=loss
    loss.backward()  
    
    optimizer.step()
  print(f'loss in epoch number {i+1}  is equal to {SumLoss} ')


def checkAccuracy(dataLoader,model):
   if dataLoader.dataset.train :
      print(' Accuracy on train data ')
   else:
      print(' Accuracy on test data ')

   num_correct=0
   num_sample=0

   model.eval() 
  
   with torch.no_grad():
      for x,y in dataLoader:
            x=x.to(device)
            y=y.to(device)

            x=x.reshape(x.shape[0],-1)
            score=model(x)
            _,pred=score.max(1) 
            num_correct+=(pred==y).sum() 
            num_sample+=len(y)
      print(f'accuracy is equal to {num_correct/num_sample}')

   model.train()

#accuaracy
checkAccuracy(train_dl,model)
checkAccuracy(test_dl,model) 

