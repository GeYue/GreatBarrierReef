#coding=UTF-8

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile
from sklearn.model_selection import GroupKFold

FOLD      = 4 # which fold to train
DIM       = 3000
MODEL     = 'yolov5s6'
BATCH     = 4
EPOCHS    = 7
OPTMIZER  = 'Adam'

PROJECT   = 'great-barrier-reef-public' # w&b in yolov5
NAME      = f'{MODEL}-dim{DIM}-fold{FOLD}' # w&b for yolov5

REMOVE_NOBBOX = True # remove images with no bbox
ROOT_DIR  = '/home/xyb/Project/Kaggle/GreatBarrierReef'
IMAGE_DIR = '/home/xyb/Project/Kaggle/GreatBarrierReef/yolo_data_all/train/images' # directory to save images
LABEL_DIR = '/home/xyb/Project/Kaggle/GreatBarrierReef/yolo_data_all/train/labels' # directory to save labels

df = pd.read_csv("./input/train.csv")
df["num_bbox"] = df['annotations'].apply(lambda x: str.count(x, 'x'))
df['image_path']  = f'{IMAGE_DIR}/'+df.image_id+'.jpg'
df['label_path']  = f'{LABEL_DIR}/'+df.image_id+'.txt'
df_train = df

## Remove all empty dataset 
#df_train = df.query("num_bbox>0")

kf = GroupKFold(n_splits = 5)
df_train = df_train.reset_index(drop=True)
df_train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y = df_train.video_id.tolist(), groups=df_train.sequence)):
    df_train.loc[val_idx, 'fold'] = fold

train_files = []
val_files   = []
train_df = df_train.query("fold!=@FOLD")
valid_df = df_train.query("fold==@FOLD")

## ONLY Remove all empty dataset from validation sets 
valid_df = valid_df.query("num_bbox>0")

train_files += list(train_df.image_path.unique())
val_files += list(valid_df.image_path.unique())
print ("training_set_number=={}, validation_set_number={}, total_num=={}".format(len(train_files), len(val_files), len(train_files)+len(val_files)))

import yaml
cwd = ROOT_DIR + "/GBR_data_set_txt/"

train_file_name = "train-" + str(FOLD) + ".txt"
val_file_name = "val-" + str(FOLD) + ".txt"
yaml_file_name = "gbr-" + str(FOLD) + ".yaml"

with open(os.path.join(cwd , train_file_name), 'w') as f:
    for path in train_df.image_path.tolist():
        f.write(path+'\n')

with open(os.path.join(cwd , val_file_name), 'w') as f:
    for path in valid_df.image_path.tolist():
        f.write(path+'\n')

data = dict(
    path  = cwd,
    train =  os.path.join(cwd , train_file_name) ,
    val   =  os.path.join(cwd , val_file_name),
    nc    = 1,
    names = ['cots'],
    )

with open(os.path.join(cwd , yaml_file_name), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(os.path.join(cwd , yaml_file_name), 'r')
print('\nyaml:')
print(f.read())
