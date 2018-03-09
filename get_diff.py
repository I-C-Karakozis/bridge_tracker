import argparse
import cv2
import numpy as np
import os

from tools.general_purpose import *
from tools import gt, imeditor

def main(args):

    # load directories
    imgs1 = collect_all_files(args.dir1)
    imgs2 = collect_all_files(args.dir2)
    for img1_file in imgs1:
        for img2_file in imgs2:

            # load images
            img1 = cv2.imread(img1_file)
            img1 = gt.extract_card(img1)
            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
            imeditor.label_pixels(img1, img1_file[-5])
            img2 = cv2.cvtColor(cv2.imread(img2_file),cv2.COLOR_BGR2GRAY)
            dims = np.shape(img1)
            diff = np.zeros(dims)

            # compute diff
            for row in range(dims[0]):
                for col in range(dims[1]):
                    if img1[row][col] == img2[row][col]:
                        if img1[row][col] == 255:
                            diff[row][col] = 255
                        else:
                            diff[row][col] = 0
                    else:
                        diff[row][col] = 60

            # show diff
            print(img1_file)
            print(img2_file)
            cv2.imshow("Diff", diff)
            key = cv2.waitKey(100000) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()

'''
Sample execution: 
python get_diff.py dir1 dir2 
'''
DESCRIPTION = """Show the diff between the images in the two directories."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    args = parser.parse_args()
    main(args)
