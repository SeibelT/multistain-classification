import numpy as np 
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

folderpath = 'data/'
n_mod = 3
n_patients = 20
n_tiles_per_patient = 100
tilesize = 256
for i in range(n_mod):
    path = folderpath + f"modality_{i}"
    if not os.path.exists(path):
        os.mkdir(path)
rows = [] 
for patient in range(n_patients):
    for idx in range(n_tiles_per_patient):
        label = "A" if patient%2  else "B" 
        sett = "TRAIN" if patient<int(0.7*n_patients) else "TEST" if patient<int(0.9*n_patients) else "VALIDATION"
        row = [f"pid{patient}",label,sett]
        for mod in range(n_mod):
            path = folderpath + f"modality_{mod}" 
            path += f"/pid{patient}_mod{mod}_{idx}.jpg"
            
            row.append(path)
            #rand_image = np.random.randint(0,255,(tilesize,tilesize))
            #matplotlib.image.imsave(path, rand_image)
        rows.append(row)


df = pd.DataFrame(rows)
columns = ["Patient_ID","Label","Set"]
columns+= [ f"Modality{i}_Path" for i in range(len(df.keys().tolist())-3) ]
columns
df.columns = columns
df

df.to_csv("new.csv",index=False)