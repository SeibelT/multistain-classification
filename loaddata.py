import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class MultiModalLoader(Dataset):
    def __init__(self, df_path, traintest,n_mods,transform = True,shuffle = True):
        # TODO transformation 
        if shuffle:
            df = pd.read_csv(df_path).sample(frac=0.3, random_state = 1) # TODO frac=0.3 -> =1
        else:
            df = pd.read_csv(df_path)

        # Encode Categories
        self.l_enc = LabelEncoder()
        self.enc = OneHotEncoder()
        encodings = self.l_enc.fit_transform(df["Label"].unique())
        self.enc.fit(encodings.reshape(-1,1))

        self.n_mods = n_mods
        # Train,Test,Validation split 
        if traintest == "TRAIN":
            self.df = df[df["Set"]=="TRAIN"]
        elif traintest == "TEST":
            self.df =df[df["Set"]=="TEST"]
        elif traintest == "VALIDATION":
            self.df = df[df["Set"]=="VALIDATION"]

        

        

        if transform:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.ColorJitter(),
            transforms.RandomRotation(degrees=180),
                                 ])
    def __len__(self):
        # Return the length of your data
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx,1 ]
        label = self.l_enc.transform([label])
        label = self.enc.transform([label]).toarray()
        fpaths = self.df.iloc[idx,-self.n_mods:]
        
        imgs = [self.transform(Image.open(fpath)) for fpath in fpaths ]

        return torch.stack(imgs),torch.tensor(label).squeeze()

    def __reverse_transformation__(self,data):
        out = []
        for x in data.detach().numpy(): 
            x = self.enc.inverse_transform([x])
            x = self.l_enc.inverse_transform(x[0])
            out.append(x.item())
        return out 

def uni_traintest_df(slidename,clininame,blocks,trainsplit,label="isMSIH"):
   """returns train and test dataframes stratified by patient IDs
      containing FILENAME and label per tile.
      Also returns weights for each class"""
   df_clini = pd.read_excel(clininame)[['PATIENT',"TCGA Project Code",label]]
   df_slide = pd.read_csv(slidename)

   tiles_exist = [] #names of tiles in data 
   for root, dirs, files in os.walk(blocks, topdown=False):
      for name in files:
         #print(os.path.join(root, name))
         tiles_exist.append(name.split(".")[0])

   df_slide = df_slide[df_slide["FILENAME"].isin(set(tiles_exist))] # patients with available tiles
   whole_df = df_slide.join(df_clini.set_index("PATIENT"), on="PATIENT")
   whole_df = whole_df[whole_df["TCGA Project Code"].notna()] # patient with available rows in slide and clini

   #stratify patients to test and train sets
   split = int(len(whole_df)*trainsplit)
   test_df = whole_df.iloc[:split,:]
   train_df  = whole_df.iloc[split:,:]
   weights = (test_df['isMSIH'].value_counts()/len(test_df['isMSIH'])).to_dict() # calc weights 

   return train_df,test_df,weights

#For one cohort 
#slidename = "data/modality1/TCGA-CRC-DX_SLIDE.csv"
#clininame ="data/TCGA-CRC-DX_CLINI.xlsx"
#blocks = "data/modality1/tcga_features_xiyuewang/"
#trainsplit = 0.75
#train_df,test_df,weights = uni_traintest_df(slidename,clininame,blocks,trainsplit)