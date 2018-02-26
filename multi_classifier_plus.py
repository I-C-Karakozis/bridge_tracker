import argparse
import cv2
import numpy as np
import os

from tools import Cards, imeditor
from tools.general_purpose import *

def main(args):

    # load groundtruth images and their labels
    path = os.path.dirname(os.path.abspath(__file__))
    gt_labels = os.listdir(os.path.join(path, "gt"))
    gt_imgs = load_imgs_gr("gt", gt_labels)
    vfunc = np.vectorize(lambda t: t[0:2])
    gt_labels = vfunc(gt_labels)

    # collect all testing data filenames
    data_filenames = []
    for path, subdirs, files in os.walk(args.data_dir):
        for name in files:
            data_filenames.append(os.path.join(path, name))

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

            # For each contour detected:
            for i in range(len(cnts_sort)):
                if (cnt_is_card[i] == 1):
                    # Create a card object from the contour and append it to the list of cards.
                    # preprocess_card function takes the card contour and contour and
                    # determines the cards properties (corner points, etc). It generates a
                    # flattened 200x300 image of the card, and isolates the card's
                    # suit and rank from the image.
                    cards.append(Cards.preprocess_card(cnts_sort[i],image))

                    # Find the best rank and suit match for the card.
                    Cards.match_card_x(cards[k], gt_labels, gt_imgs)
                    cards[k].best_match, cards[k].diff = Cards.match_card_x(cards[k],gt_labels,gt_imgs)

                    # Draw center point and match result on the image.
                    image = Cards.draw_results(image, cards[k])
                    k = k + 1
            
            # Draw card contours on image (have to do contours all at once or
            # they do not show up properly for some reason)
            if (len(cards) != 0):
                temp_cnts = []
                for i in range(len(cards)):
                    temp_cnts.append(cards[i].contour)
                cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
        
        cv2.imwrite(os.path.join(args.target_dir, '{:06d}.png'.format(imgs)), image)
        imgs = imgs + 1
        # cv2.imshow("Card Detector",image)
        # key = cv2.waitKey(100000) & 0xFF
        # if key == ord("q"):
        #     return

    return

'''
Sample execution: 
python multi_classifier_x.py data_dir target_dir 
'''
DESCRIPTION = """Multi-classifier of playing card suit-value. Requires cards placed on green felt."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('data_dir', help='Directory of testing data.')
    parser.add_argument('target_dir', help='Directory to store classified images in.')
    args = parser.parse_args()
    main(args)
