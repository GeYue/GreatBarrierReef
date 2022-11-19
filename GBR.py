#coding=UTF-8

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shutil import copyfile

train = pd.read_csv('./input/train.csv')
train['pos'] = train.annotations != '[]'

for idx, x in train.iterrows():
    if not x.pos: continue
    copyfile(f'./input/train_images/video_{x.video_id}/{x.video_frame}.jpg', 
            f'./yolo_data/train/images/{x.image_id}.jpg')
    anno = eval(x.annotations)
    r = ''
    for an in anno:
        r += '0 {} {} {} {}\n'.format((an['x'] + an['width'] / 2) / 1280,
                                      (an['y'] + an['height'] / 2) / 720,
                                      an['width'] / 1280, 
                                      an['height'] / 720)
    path = os.getcwd()
    with open(f'./yolo_data/train/lables/{x.image_id}.txt', 'a') as fp:
        fp.write(r)
    val = 1

