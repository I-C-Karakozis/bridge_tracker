import argparse
import cv2
import random
import os

from tools import Cards, imeditor, gt
from tools.general_purpose import *

def main(args):

    # collect all data filenames
    data_filenames = []
    for path, subdirs, files in os.walk(args.data_dir):
        for name in files:
            data_filenames.append(os.path.join(path, name))

    setup_dir(args.target_dir)

    train = 0
    val = 0

    for img_file in data_filenames:
        # read image and get its gt label
        print(img_file)
        image = cv2.imread(img_file)
        gt_label = img_file.split('/')[-2]
        name = img_file.split('/')[-1][:-4]
        
        # save image in flattened dir so that p*N of the images end up in train directory
        v = random.uniform(0.0, 1.0)
        if args.p > v:
            target = os.path.join(args.target_dir, 'train')
            cv2.imwrite(os.path.join(target, name+'_'+gt_label+'.png'), image)
            train = train + 1
        else:
            target = os.path.join(args.target_dir, 'val')
            cv2.imwrite(os.path.join(target, name+'_'+gt_label+'.png'), image)
            val = val + 1

    # report actual distirbution
    print("Training:", train, train / (train+val))
    print("Validation:", val, val / (train+val))

'''
Sample execution: 
python flatten_dir.py data_dir target_dir p
'''
DESCRIPTION = """Splits data in data_dir into training and validation dataset where p=train/total."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('data_dir', help='Directory with data.')
    parser.add_argument('target_dir', help='Directory to save images in.')
    parser.add_argument('p', type=float, help='Proportion of data to be allocated to training. Pick value in (0,1)')
    args = parser.parse_args()
    main(args)
