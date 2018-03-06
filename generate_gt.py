import argparse
import cv2
import imutils
import numpy as np
import os

from tools import Cards, gt, imeditor
from tools.general_purpose import *

def main(args):

    # collect all testing data filenames
    data_filenames = collect_all_files(args.data_dir)    

    imgs = 1
    for img_file in data_filenames:
        # load image
        print(imgs, "-", img_file)
        image = cv2.imread(img_file)
        gt_name = img_file.split('/')[-2]
        suit = gt_name[1]

        # store ground truth
        gt_raw = gt.extract_card(image)
        if gt_raw is not None:
            gt_labelled = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2GRAY) 
            imeditor.label_pixels(gt_labelled, suit)
            cv2.imwrite(os.path.join(args.target_dir, gt_name+".png"), gt_labelled)

            # store inverted ground truth
            gt_labelled_inv = imutils.rotate_bound(gt_labelled, 180)
            cv2.imwrite(os.path.join(args.target_dir, gt_name+"_inv"+".png"), gt_labelled_inv)
        
        imgs = imgs + 1

'''
Sample execution: 
python generate_gt.py data_dir target_dir 
'''
DESCRIPTION = """Form ground truth detections from images in data_dir 
                 and saves them in target_dir."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('data_dir', help='Directory of training data.')
    parser.add_argument('target_dir', help='Directory to store classified images in.')
    args = parser.parse_args()
    main(args)
