import argparse
import cv2
import numpy as np
import os

import imutils
import skvideo.io
import time

### ---- INITIALIZATION ---- ###

FPS = 24

### ------------------------ ###


## add mogrify in script
## mogrify -format JPG Images/002/*.jpg

def main(args):

    start = time.time()

    # load video
    cap = skvideo.io.vreader(args.video)
    frame = next(cap, None)
    dims = np.shape(frame)
    count = 1

    while frame is not None:
        if count % 100 == 0:
            print('Reading frame: ', count)
            print('Time elapsed: ', time.time() - start)

        # store 1 frame per second
        if count % FPS == 0:
            # fix frame orientation
            rot_frame = imutils.rotate_bound(frame, args.angle)
            # cv2.imshow("Frame", frame)
            # cv2.imshow("Rot_Frame", rot_frame)
            # key = cv2.waitKey(100000) & 0xFF
            # if key == ord("q"):
            #     cv2.destroyAllWindows()
            #     return

            # save frame
            skvideo.io.vwrite(os.path.join(args.target_dir, "second{:06d}.jpg".format(count)), rot_frame)

        # extract one frame at a time
        frame = next(cap, None)
        count = count +1

'''
Sample execution: 
python vid_to_frames.py video target_dir 45
'''
DESCRIPTION = """Exctracts 1 frame per second from video."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('video', help='Path to video.')
    parser.add_argument('target_dir', help='Directory to save frames in.')
    parser.add_argument('angle', help='Rotation angle needed to orient the table parallel to the camera.', type=int)
    args = parser.parse_args()
    main(args)
