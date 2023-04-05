
import torch
from torch import nn
import torchvision.models as models

class MultistainModel(nn.Module):
    def __init__(self,n_mods =3,d_model = 512,n_classes = 2, num_layers = 3, pretrained_models=None,pretrained_model_dicts=None,model_output_sizes=None):
        super(MultistainModel,self).__init__()
        self.d_model = d_model
        self.feat_extractors = nn.ModuleList([])
        self.n_classes = n_classes
        
            
        if pretrained_models == None: 
            for i in range(n_mods):
                model =  models.resnet18(pretrained=False)
                model = nn.Sequential(model,nn.Linear(1000,self.d_model))
                self.feat_extractors.append(model)
        else:
            for i in range(n_mods):
                pretrained_model =  pretrained_models[i]
                pretrained_model.requires_grad_(False)
                pretrained_model.load_state_dict(torch.load(pretrained_model_dicts[i]))
                pretrained_model = nn.Sequential(pretrained_model,nn.Linear(model_output_sizes[i],self.d_model))
                self.feat_extractors.append(pretrained_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.globalpooling = nn.AdaptiveAvgPool2d((1,self.d_model))
        self.linear_out = nn.Linear(self.d_model,n_classes)
        self.activate = nn.Softmax(dim=1)

        
    def forward(self,imgs):  
        B,n_mod,C,H,W = imgs.size()
        
        imgs = imgs.split(1,dim=1)
        features = []
        for idx,img in enumerate(imgs):
            features.append(self.feat_extractors[idx](img[:,0,:,:,:]))
        features = torch.stack(features,dim = 1 )
        features = self.transformer_encoder(features)
        features = self.globalpooling(features)
        features = self.linear_out(features) 
        #output = self.activate(features)
        
        return features.view((B,self.n_classes))
