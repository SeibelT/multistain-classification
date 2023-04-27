import os
import sys
import torch 
import pandas as pd
import time 
from torch import nn
from torch import optim
from torchmetrics import AUROC
from torch.utils.data import DataLoader

from data_loader.loaddata import MultiModalLoader
from model.model import MultistainModel
from trainer.train import trainer
from eval.evaluate import evaluation

from torch.utils.tensorboard import SummaryWriter
start = time.time()
###Variables 
expname = "/ex1"
table_path = "./data/datatable.csv"
n_mods = 3
n_epochs = 2
train_bs = 64
n_classes = 2 #TODO still hardcoded for 2 only 
unfreeze_epoch = 1
weighted = True 
zip_path = "/projects/p_scads_pathology/MultiStain/tiles.zip" # if data not in zip file  zip_path = None
###



# create folder to store results
filepath= "./results"+expname
try:
  os.mkdir(filepath)
except FileExistsError:
  print("Path already exists")
  #sys.exit()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device.type}")

train_DS = MultiModalLoader(table_path, "TRAIN",n_mods=n_mods,zip_path=zip_path)
test_DS = MultiModalLoader(table_path, "TEST",n_mods=n_mods,zip_path=zip_path)
valid_DS = MultiModalLoader(table_path, "VALIDATION",n_mods=n_mods,zip_path=zip_path)

train_loader = DataLoader(train_DS, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(valid_DS, batch_size=32, shuffle=False)
test_loader = DataLoader(test_DS, batch_size=32, shuffle=False)


##Tensorboard writer 
writer = SummaryWriter((filepath+"/tensorboard"))

model = MultistainModel(n_classes = n_classes)
model.to(device)

#model = torch.compile(premodel)  # TODO needs torch 2.0 and GPU 

if weighted:
  criterion = nn.CrossEntropyLoss(weight=train_DS.__getweights__())
else:
  criterion = nn.CrossEntropyLoss()  

optimizer = optim.Adam(model.parameters(), lr=0.001)  # TODO optimal lr for this task?
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9) 
auroc = AUROC(task="multilabel", num_labels=n_classes)  # TODO do for categorical
activate = nn.Softmax(dim=1) 

##Network Graph
inputs = torch.randn(1,3,3,224, 224)
inputs = inputs.to(device).float()
writer.add_graph(model, inputs)

##Train Model
model,optimizer,epoch = trainer(model,n_epochs,train_loader,valid_loader,scheduler,device,criterion,optimizer,writer,auroc=auroc,activate=activate,unfreeze=unfreeze_epoch,checkpoint_path = filepath)


writer.close()

##Final saving of weights 
state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    }

torch.save(state, filepath+"/model.pt")


##Evaluation
results = evaluation(model,test_loader,device,criterion,auroc,activate)

results_df = pd.DataFrame(results,columns=["cls1_label","cls2_label","cls1_pred","cls2_pred"])  # TODO hardcoded for 2 classes only
        

f = results_df.to_csv((filepath+"/result_table.csv"),index=False)

print(f"Total time: {time.time-start}s")