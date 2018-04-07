import argparse
import cv2
import numpy as np
import os
import time

from tools import Cards, gt, imeditor
from tools.general_purpose import *

def main(args):

    # load groundtruth images and their labels
    path = os.path.dirname(os.path.abspath(__file__))
    gt_labels = os.listdir(os.path.join(path, args.gt_dir))
    gt_imgs = load_imgs_gr(args.gt_dir, gt_labels)
    vfunc = np.vectorize(lambda t: t[0:2])
    gt_labels = vfunc(gt_labels)

    # collect all testing data filenames
    data_filenames = []
    for path, subdirs, files in os.walk(args.data_dir):
        for name in files:
            data_filenames.append(os.path.join(path, name))

    # start timing
    start = time.time()

    # localize and classify all cards on each image
    imgs = 0
    errors = 0.0
    for img_file in data_filenames:
        # read image and get its gt label
        print(imgs, "-", img_file)
        image = cv2.imread(img_file)
        gt_label = img_file.split('/')[-2]

        # find and classify all cards in image
        label = gt.find_cards(image, gt_labels, gt_imgs)        

        if label is None or label != gt_label:
            errors = errors + 1.0
            print("Mistake:", label)

            # debugging mode --> show warps
            if args.show_errors != 0:
                cv2.imshow("Gray", cards[0].warp)
                key = cv2.waitKey(100000) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()            
        
        cv2.imwrite(os.path.join(args.target_dir, '{:06d}.png'.format(imgs)), image)
        imgs = imgs + 1

    # report timing metrics
    end = time.time()
    print("Total Classification Time:", end - start)
    print("Time per Classification:", (end - start) / len(data_filenames))

    # report performance metrics
    print("Accuracy:", 1 - errors / len(data_filenames))
    print("Misclassification Count:", errors)

    return

'''
Sample execution: 
python multi_classifier.py gt data/training_data data/classifications 0
'''
DESCRIPTION = """Multi-classifier of playing card suit-value. Requires cards placed on green felt."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('gt_dir', help='Directory of groundtruth data.')
    parser.add_argument('data_dir', help='Directory of testing data.')
    parser.add_argument('target_dir', help='Directory to store classified images in.')
    parser.add_argument('show_errors', type=int, help='Enter 0 to disable debugging mode')
    args = parser.parse_args()
    main(args)
