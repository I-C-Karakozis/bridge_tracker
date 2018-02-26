import argparse
import cv2
import numpy as np
import os

import Cards
import stats

### ---- INITIALIZATION ---- ###

# confidence level for background/foreground predictions
CONFIDENCE = 99

# we use 3 degrees of freedom bc we use the RGB color space
DOF = 3

### ------------------------ ###

def model_boundary(img):
    ''' 
    Model the color distribution of the boundary pixels as 
    a Multivariate Gaussian with diagonal covariance matrix
    '''

    # compute boundary size from image dimensions
    dims = np.shape(img)
    boundary_size = 2*dims[0] + 2 * dims[1]
    boundary = np.zeros((boundary_size, 3))
    count = 0

    # collect pixels of leftmost and rightmost image boundaries
    for col in [0, dims[1]-1]:
        for row in range(dims[0]):
            boundary[count] = img[row][col]
            count = count + 1

    # collect pixels of top and bottom image boundaries
    for row in [0, dims[0]-1]:
        for col in range(dims[1]):
            boundary[count] = img[row][col]
            count = count + 1

    # compute multivariate gaussian 
    mu = np.mean(boundary, axis=0)
    sigmas = np.std(boundary, axis=0)
    return mu, sigmas

def remove_background(img, mu, sigmas):
    ''' 
    Remove form img background pixels whose color distirbution is modeled
    by Multivariate Gaussian with mean mu and covariance matrix diag(sigmas^2).
    '''
    dims = np.shape(img)

    zscore = np.square((img - mu).astype(np.float) / sigmas)
    zscore = np.reshape(np.apply_over_axes(np.sum, zscore, [2]), (dims[0], dims[1]))
    zscore = np.repeat(zscore[:, :, np.newaxis], 3, axis=2)
    mask = np.where(stats.h_test(zscore, confidence=CONFIDENCE, 
                                       dof=DOF), 0, 255).astype(np.uint8)

    black_frame = np.zeros(dims, dtype=np.uint8)
    foreground = np.where(mask > 0, img, black_frame)
    return foreground

def correct_img_dim(image):
    '''Adjust/Rotate image into 720x1280 (height by width). '''

    # rotate image so that height < width
    dims = np.shape(image)
    if dims[0] > dims[1]:
        image = np.transpose(image)

    # resize image into (IM_HEIGHT x IM_WIDTH)
    corrected = cv2.resize(image, (IM_WIDTH, IM_HEIGHT))
    return corrected

def main(args):

    # Load the train rank and suit images
    path = os.path.dirname(os.path.abspath(__file__))
    train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
    train_suits = Cards.load_suits( path + '/Card_Imgs/')

    # remove background felt from image
    image = cv2.imread(args.test_img)
    green_mu, green_sigmas = model_boundary(image)
    no_felt = remove_background(image, green_mu, green_sigmas)

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
python preprocess.py raw_data_dir target_dir 
'''
DESCRIPTION = """Baseline classifier of playing card suit-value. Requires cards placed on green felt."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('test_img', help='Directory with raw images.')
    # parser.add_argument('target_dir', help='Directory to store adjusted images in.')
    args = parser.parse_args()
    main(args)
