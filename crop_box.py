import argparse
import cv2
import os

from tools import Cards, imeditor, gt
from tools.general_purpose import *

def main(args):

    # load groundtruth frames and their labels
    path = os.path.dirname(os.path.abspath(__file__))
    gt_labels = os.listdir(os.path.join(path, args.gt_dir))
    gt_imgs = load_imgs_gr(args.gt_dir, gt_labels)
    vfunc = np.vectorize(lambda t: t[0:2])
    gt_labels = vfunc(gt_labels)

    # load all frames and their bounding boxes
    frames_files = collect_all_files(args.frames_dir)
    boxes_files = collect_all_files(args.box_dir)

    # measure dataset size
    box_count = 0
    frame_count = len(frames_files)

    for frame, box_txt in zip(frames_files, boxes_files):
        # load frame and ground truth
        print(frame)
        frame = cv2.imread(frame)
        with open(box_txt) as f:
            boxes = f.readlines()
        print("Boxes:", boxes[0])
        boxes = boxes[1:]

        # draw bounding boxes on frame
        for box in boxes:
            # load coordinates
            xy = box.split()
            label = xy[-1]
            a = 2.5
            xy = [int(2.5*int(x)) for x in xy[:-1]]

            # get card crop
            crop = frame[xy[1]:xy[3], xy[0]:xy[2]]

            if crop is not None and label != "Dummy":
                box_count = box_count + 1
                # show crop
                # cv2.imshow("Boxes", crop)
                # key = cv2.waitKey(100000) & 0xFF
                # if key == ord("q"):
                #     cv2.destroyAllWindows()

                # classify
                pred = gt.find_cards(crop, gt_labels, gt_imgs, debug=0)

            # draw box
            cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), (255,255,255), thickness=4)

        # show frame
        cv2.imshow("Boxes", frame)
        key = cv2.waitKey(100000) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()

        print("Frames:", frame_count, "- Boxes:", box_count)

'''
Sample execution: 
python crop_box.py frames_dir box_dir gt_dir
'''
DESCRIPTION = """Classifies cards in bounding boxes."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('frames_dir', help='Directory with frames to classify cards from.')
    parser.add_argument('box_dir', help='Directory with text documents with bounding boxes coordinates.')
    parser.add_argument('gt_dir', help='Directory with groundtruth templates.')
    args = parser.parse_args()
    main(args)
