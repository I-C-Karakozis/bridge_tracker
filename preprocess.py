import argparse
import cv2
import numpy as np
import os

### ---- INITIALIZATION ---- ###

# 1080p dimensions
WIDTH = 1920.0
HEIGHT = 1080.0

### ------------------------ ###

def adjust_dim(image, width):
    '''Resize width of image into width but maintain aspect ratio.'''

    # rotate image so that height < width
    dims = np.shape(image)
    if dims[0] > dims[1]:
        image = np.transpose(image)

    # resize image but maintain aspect ratio
    dims = np.shape(image)
    height = (WIDTH / dims[1]) * dims[0]
    corrected = cv2.resize(image, (int(width), int(height)))

    return corrected

### ------------------------ ###
def main(args):
    # input directory setup
    print("Source:", args.raw_data_dir)
    raw_imgs = os.listdir(args.raw_data_dir)

    # output directory setup
    print("Dest:", args.target_dir)
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    # standardize all images to 1080p dimensions
    count = 0
    for rimg_name in raw_imgs:
        print(count)
        rimg = cv2.imread(os.path.join(args.raw_data_dir, rimg_name))
        print(np.shape(rimg))
        adjusted = adjust_dim(rimg, WIDTH)
        cv2.imwrite(os.path.join(args.target_dir, '{:06d}.png'.format(count)), adjusted)
        count = count + 1

'''
Sample execution: 
python preprocess.py raw_data_dir target_dir 
'''
DESCRIPTION = """Preprocess images in raw_data_dir and store the preprocessed result in target_dir."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('raw_data_dir', help='Directory with raw images.')
    parser.add_argument('target_dir', help='Directory to store adjusted images in.')
    args = parser.parse_args()
    main(args)
