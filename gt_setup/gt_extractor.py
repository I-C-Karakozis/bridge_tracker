import argparse
import cv2
import numpy as np
import os

from tools import Cards, imeditor
from tools.general_purpose import *

def main(args):

    # collect all testing data filenames
    data_filenames = collect_all_files(args.data_dir)    

    # remove background felt from image
    imgs = 0
    for img_file in data_filenames:
        print(imgs, "-", img_file)
        image = cv2.imread(img_file)

        # Remove felt background
        green_mu, green_sigmas = imeditor.model_boundary(image)
        no_felt = imeditor.remove_background(image, green_mu, green_sigmas)

        # Pre-process camera image (gray, blur, and threshold it)
        pre_proc = Cards.preprocess_image(no_felt)
        
        # Find and sort the contours of all cards in the image (query cards)
        cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

        # If there are no contours, do nothing
        if len(cnts_sort) != 0:
            # Initialize a new "cards" list to assign the card objects.
            # k indexes the newly made array of cards.
            cards = []
            k = 0

            # For each card contour detected, extract card
            for i in range(len(cnts_sort)):
                if (cnt_is_card[i] == 1):
                    print("found")
                    card = Cards.extract_card(cnts_sort[i], image)
                    cv2.imwrite(os.path.join(args.target_dir, '{:06d}.png'.format(imgs)), card)
                    imgs = imgs + 1

'''
Sample execution: 
python gt_extractor.py data_dir target_dir 
'''
DESCRIPTION = """Form ground truth detections from images in data_dir 
                 and saves them in target_dir."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('data_dir', help='Directory of training data.')
    parser.add_argument('target_dir', help='Directory to store classified images in.')
    args = parser.parse_args()
    main(args)
