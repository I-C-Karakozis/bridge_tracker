import cv2
import numpy as np
import os

import Cards
import imeditor

def extract_card(image):
    ''' Extract a single card from image; 
        Return None if none exists'''

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
                card = Cards.extract_card(cnts_sort[i], image)
                return card

    return None

def find_cards(image, gt_labels, gt_imgs, debug=0):
    '''
    Finds all cards in image and identifies their rank and suit 
    from the gt_labels and the gt_imgs (groundtruth templates);
    If debug=1, it shows the cards warps created before classification.
    '''

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
        label = "No card found."

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

                # show warps
                if debug == 1:
                    cv2.imshow("Warp", cards[k].color_warp)
                    key = cv2.waitKey(100000) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()

                # Draw center point and match result on the image.
                image, label = Cards.draw_results(image, cards[k])
                k = k + 1 

        # Draw card contours on image for all cards
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)  

        return label

    return None
