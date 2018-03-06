import argparse
import cv2
import numpy as np
import os
import time

from tools import Cards, imeditor
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
        gt = img_file.split('/')[-2]

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
                    cards[k].best_match, cards[k].diff = Cards.match_card(cards[k],gt_labels,gt_imgs)

                    # Draw center point and match result on the image.
                    image, label = Cards.draw_results(image, cards[k])
                    k = k + 1

                    if label != gt:
                        errors = errors + 1.0
                        print("Mistake:", label)

                        # debugging mode: show warps
                        if args.show_errors != 0:
                            cv2.imshow("Gray",cards[0].warp)
                            cv2.imshow("Color",cards[0].color_warp)
                            key = cv2.waitKey(100000) & 0xFF
                            if key == ord("q"):
                                cv2.destroyAllWindows()
            
            # Draw card contours on image (have to do contours all at once or
            # they do not show up properly for some reason)
            if (len(cards) != 0):
                temp_cnts = []
                for i in range(len(cards)):
                    temp_cnts.append(cards[i].contour)
                cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
        
        cv2.imwrite(os.path.join(args.target_dir, '{:06d}.png'.format(imgs)), image)
        imgs = imgs + 1

    # report timing metrics
    end = time.time()
    print("Classification Time:", end - start)
    print("Time per Classification:", (end - start) / len(data_filenames))

    # report performance metrics
    print("Accuracy:", 1 - errors / len(data_filenames))
    print("Misclassification Count:", errors)

    return

'''
Sample execution: 
python multi_classifier.py gt_dir data_dir target_dir 0
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
