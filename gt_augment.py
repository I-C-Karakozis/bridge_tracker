import argparse
import cv2
import imutils
import numpy as np
import os

from tools import Cards, imeditor
from tools.general_purpose import *

def main(args):

    # collect all testing data filenames
    gt_filenames = collect_all_files(args.gt_dir)    

    imgs = 0
    for gt in gt_filenames:
        label = gt.split('/')[1][0:2]
        gt_image = cv2.imread(gt)
        gt_image = cv2.cvtColor(gt_image,cv2.COLOR_BGR2GRAY)

        # rotate and warp
        rows, cols = np.shape(gt_image)
        dst = imutils.rotate_bound(gt_image, 90)
        augm = cv2.resize(dst, (cols, rows))

        cv2.imwrite(os.path.join(args.target_dir, label+('{:03d}.png'.format(imgs))), augm)
        imgs = imgs + 1

    return

'''
Sample execution: 
python gt_augmnet.py gt_dir target_dir 
'''
DESCRIPTION = """Augment ground truth images by considering horizontal warping them
                 to a horizontal perspective."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('gt_dir', help='Directory of groundtruth data.')
    parser.add_argument('target_dir', help='Directory to store classified images in.')
    args = parser.parse_args()
    main(args)
