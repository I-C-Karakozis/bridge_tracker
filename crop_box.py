import argparse
import cv2
import os

from tools import Cards, imeditor, gt
from tools.general_purpose import *

def main(args):

    # load groundtruth images and their labels
    path = os.path.dirname(os.path.abspath(__file__))
    gt_labels = os.listdir(os.path.join(path, args.gt_dir))
    gt_imgs = load_imgs_gr(args.gt_dir, gt_labels)
    vfunc = np.vectorize(lambda t: t[0:2])
    gt_labels = vfunc(gt_labels)

    # load image and ground truth
    image = cv2.imread(args.image)
    with open(args.box_txt) as f:
        boxes = f.readlines()
    boxes = boxes[1:]

    # draw bounding boxes on image
    for box in boxes:
        # load coordinates
        xy = box.split()
        label = xy[-1]
        xy = [2*int(x) for x in xy[:-1]]

        # get card crop
        crop = image[xy[1]:xy[3], xy[0]:xy[2]]

        if crop is not None:
            # show crop
            cv2.imshow("Boxes", crop)
            key = cv2.waitKey(100000) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()

            # classify
            pred = gt.find_cards(crop, gt_labels, gt_imgs, debug=1)
            # print("Prediction:", pred)
            # print("Groundtruth:", label)

    # show image
    cv2.imshow("Boxes", image)
    key = cv2.waitKey(100000) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()

'''
Sample execution: 
python crop_box.py image box_txt gt_dir
'''
DESCRIPTION = """Classifies cards in bounding boxes."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('image', help='Image to classify cards from.')
    parser.add_argument('box_txt', help='Text document with bounding boxes.')
    parser.add_argument('gt_dir', help='Directory with groundtruth templates.')
    args = parser.parse_args()
    main(args)
