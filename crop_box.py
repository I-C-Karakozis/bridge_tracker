import argparse
import cv2
import os

from tools import Cards, imeditor
from tools.general_purpose import *

def classify(image, gt_labels, gt_imgs):
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
                key = cv2.waitKey(100000) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()

                # Draw center point and match result on the image.
                image, label = Cards.draw_results(image, cards[k])
                k = k + 1   

                return label

    return None

def main(args):

    # load groundtruth images and their labels
    path = os.path.dirname(os.path.abspath(__file__))
    gt_labels = os.listdir(os.path.join(path, args.gt_dir))
    gt_imgs = load_imgs_gr(args.gt_dir, gt_labels)
    vfunc = np.vectorize(lambda t: t[0:2])
    gt_labels = vfunc(gt_labels)

    # load image and ground truth
    image = cv2.imread(args.image)
    with open(args.gt_txt) as f:
        gt = f.readlines()
    gt = gt[1:]

    # draw bounding boxes on image
    for box in gt:
        # load coordinates
        xy = box.split()
        label = xy[-1]
        xy = [int(x) for x in xy[:-1]]

        # get card crop
        crop = image[xy[1]:xy[3], xy[0]:xy[2]]
        cv2.imshow("Boxes", crop)
        key = cv2.waitKey(100000) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()

        # classify
        pred = classify(crop, gt_labels, gt_imgs)
        print("Prediction:", pred)
        print("Groundtruth:", label)

    # show image
    cv2.imshow("Boxes", image)
    key = cv2.waitKey(100000) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()

    return

'''
Sample execution: 
python vid_to_frames.py image gt_txt gt_dir
'''
DESCRIPTION = """Exctracts 1 frame per second from video."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('image', help='Path to video.')
    parser.add_argument('gt_txt', help='Directory to save frames in.')
    parser.add_argument('gt_dir', help='Directory to save frames in.')
    args = parser.parse_args()
    main(args)
