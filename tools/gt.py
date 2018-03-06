import cv2
import numpy as np
import os

import Cards
import imeditor

def extract_card(image):
    ''' extract a single card from image; 
        return None if none exists'''

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


