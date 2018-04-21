import cv2
import numpy as np
import shutil
import os

### ---- INITIALIZATION ---- ###

BASE_DIR = "test_data"
SUITS = ["H","D","S","C"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

### ------------------------ ###

def collect_all_files(target_dir):
    filenames = []
    for path, subdirs, files in os.walk(target_dir):
        for name in files:
            filenames.append(os.path.join(path, name))

    return filenames

def load_imgs_gr(target_dir, img_filenames):
    """Loads all images in target_dir/img_filenames in grayscale."""
    imgs = []
    for l in img_filenames:
        img = cv2.imread(os.path.join(target_dir, l))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        imgs.append(img)        
    
    return imgs

def setup_dir(top_dir):
    ''' Create directory with training and validation subdirectories,
        each with a subdirectory for each class label. '''
    
    # remove old top directory
    if os.path.exists(top_dir):
        shutil.rmtree(top_dir)

    train_dir = os.path.join(top_dir, 'train')
    val_dir = os.path.join(top_dir, 'val')

    # setup up new base top directory
    os.makedirs(top_dir)
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    # setup class subdirectories
    for r in RANKS:
        for s in SUITS:
            train_class_dir = os.path.join(train_dir, r+s)
            os.makedirs(train_class_dir)
            val_class_dir = os.path.join(val_dir, r+s)
            os.makedirs(val_class_dir)            
