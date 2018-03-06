import argparse
import cv2
import numpy as np
import os

from tools.general_purpose import *
from tools import imeditor

def main(args):

    # collect all testing data filenames
    data_filenames = collect_all_files(args.gt_dir)    

    # remove background felt from image
    for img_file in data_filenames:
        # read image file
        print(img_file)
        image = cv2.imread(img_file)
        gt_name = img_file.split('/')[1]
        suit = gt_name[1]

        # label and save image
        imeditor.label_pixels(image, suit)
        cv2.imwrite(os.path.join(args.target_dir, gt_name), image)

'''
Sample execution: 
python gt_extractor.py gt_dir target_dir 
'''
DESCRIPTION = """Label every pixel of the image as white, red, or black."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('gt_dir', help='Directory of groundtruth images.')
    parser.add_argument('target_dir', help='Directory to store labelled groundtruth.')
    args = parser.parse_args()
    main(args)
