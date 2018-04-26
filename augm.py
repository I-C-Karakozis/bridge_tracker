import argparse
import cv2
import imutils
import random
import os

from tools.general_purpose import *

def main(args):

    # collect all data filenames
    data_filenames = []
    for path, subdirs, files in os.walk(args.data_dir):
        for name in files:
            data_filenames.append(os.path.join(path, name))

    old_count = len(data_filenames)
    new_count = 0

    for img_file in data_filenames:
        # read image and get its gt label
        print(img_file)
        image = cv2.imread(img_file)
        gt_label = img_file.split('/')[-2]
        name = img_file.split('/')[-1][:-4]
        
        degrees = 0
        increment = 360 / args.k

        # perform k rotations (including original); save each rotation in image directory
        for i in range(args.k-1):
            degrees = degrees + increment
            rotated = imutils.rotate_bound(image, degrees)
            target = os.path.join(args.data_dir, gt_label)
            file_name = '%s_%02d.png' % (name, degrees)
            print(file_name)
            cv2.imwrite(os.path.join(target, file_name), rotated)
            new_count = new_count + 1

    # report augmentation stats
    print("Augmented from %02d to %02d" % (old_count, new_count))

'''
Sample execution: 
python augm.py training_data 8
'''
DESCRIPTION = """Augments training dataset by performing k distinct rotations on each image."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('data_dir', help='Directory with data.')
    parser.add_argument('k', type=int, help='Number of rotations to perform; Recommend: 4 or 8.')
    args = parser.parse_args()
    main(args)
