# multistain-classification
Investigating multimodal classification for multistained Datasets. 
Inspired by 
[AGFoersch/MultiStainDeepLearning](https://github.com/AGFoersch/MultiStainDeepLearning).

The model applies pre trained feature extractors to differently stained tiles and then aggregates the feature vectors with a transformer encoder. During training, the feature extractors are frozen for the first x epochs and then jointly trained afterwards. 

The pipeline returns the trained model(model.pt), tensorboard events(within subfolder), and the prediction(result_table.csv) of the testset.


## Install
- open terminal
- create virtual environment: `python3 -m venv env`
- activate virtual environment: `source env/bin/activate`
- install requirements: `pip3 install -r requirements.txt`
- install torch: `pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu116`
``

## Settings
- put multimodal table `datatable.csv` into `multistain-classification/data` folder (name must be the same)


Patient_ID | Label | Set | Mod1_paths | Mod2_paths
---|---|---|---|---|
pid0|B|TRAIN|data/mod1/img1_mod1.jpg|data/mod2/img1_mod2.jpg



 


- put data into data folder(eg): 
```
data/
│   datatable.csv
│     
│
└───mod1
│   │
│   │   img1_mod1.jpg
│   │   img2_mod1.jpg
│   
└───mod2
│   │   img1_mod2.jpg
│   │   img2_mod2.jpg
...
```


- adapt variables at the beginning of main.py:
```
expname = "/ex1"
table_path = "data/datatable.csv"
n_mods = 3
n_epochs = 2
train_bs = 64
n_classes = 2 #TODO still hardcoded for 2 only 
unfreeze_epoch = 0
```
## Run
- open terminal 
- activate virtual environment: `source env/bin/activate`
- go into folder `cd multistain-classification`
- run main.py: `python3 main.py`

