import argparse
import cv2
import numpy as np
import os

from tools import Cards, imeditor

def main(args):

    # Load the train rank and suit images
    path = os.path.dirname(os.path.abspath(__file__))
    train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
    train_suits = Cards.load_suits( path + '/Card_Imgs/')

    # remove background felt from image
    image = cv2.imread(args.test_img)
    green_mu, green_sigmas = imeditor.model_boundary(image)
    no_felt = imeditor.remove_background(image, green_mu, green_sigmas)
    # cv2.imshow("no_felt",no_felt)
    # key = cv2.waitKey(100000) & 0xFF
    # if key == ord("q"):
    #     return

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
                print(cnts_sort[i])
                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
                cards.append(Cards.preprocess_card(cnts_sort[i],image))

                # Find the best rank and suit match for the card.
                cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

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
        
    cv2.imshow("Card Detector",image)
    key = cv2.waitKey(100000) & 0xFF
    if key == ord("q"):
        pass

    return

'''
Sample execution: 
python base_classifier.py 1080p.image 
'''
DESCRIPTION = """Baseline classifier of playing card suit-value. 
                 Requires cards placed on green felt."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('test_img', help='Directory with raw images.')
    args = parser.parse_args()
    main(args)
