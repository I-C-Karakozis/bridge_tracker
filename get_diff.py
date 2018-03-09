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
            print(img1_file)
            print(img2_file)

            # load image 1
            img1 = cv2.imread(img1_file)
            img1 = gt.extract_card(img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            imeditor.label_pixels(img1, img1_file[-5])

            # load image 2
            img2 = cv2.imread(img2_file)
            img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

            # compute diff
            dims = np.shape(img2)
            diff = np.zeros(dims + (3,))
            for row in range(dims[0]):
                for col in range(dims[1]):
                    if img1[row][col] == img2[row][col]:
                        # blue for correctly matched background
                        if img1[row][col] == 255:
                            diff[row][col] = [255, 0, 0] 
                        # green for correctly matched foreground
                        else:
                            diff[row][col] = [0, 255, 0] 
                    else:
                        # white for error
                        diff[row][col] = [255, 255, 255] 

            # show diff
            cv2.imshow("Diff", diff)
            key = cv2.waitKey(100000) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
            elif key == ord("s"):
                file_name = img1_file.split('/')[-1].split(".")[0] + "vs" + img2_file.split('/')[-1]
                ret = cv2.imwrite(os.path.join(args.target_dir, file_name), diff)
                print("Saved:", ret, file_name)
                cv2.destroyAllWindows()


'''
Sample execution: 
python get_diff.py dir1 dir2 target_dir
'''
DESCRIPTION = """Show the diff between the images in the two directories."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    parser.add_argument('target_dir')
    args = parser.parse_args()
    main(args)
