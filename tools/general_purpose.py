import cv2
import numpy as np
import os

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
