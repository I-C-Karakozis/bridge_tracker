import argparse
import cv2
import os

import skvideo.io
import time

### ---- INITIALIZATION ---- ###

FPS = 24

### ------------------------ ###

def main(args):

    start = time.time()

    # load video
    cap = skvideo.io.vreader(args.video)
    frame = next(cap, None)
    count = 1

    while frame is not None:
        if count % 100 == 0:
            print('Reading frame: ', count)
            print('Time elapsed: ', time.time() - start)

        # store 1 frame per second
        if count % FPS == 0:
            skvideo.io.vwrite(os.path.join(args.target_dir, "second{:06d}.jpg".format(count)), frame)

        # extract one frame at a time
        frame = next(cap, None)
        count = count +1

'''
Sample execution: 
python vid_to_frames.py video target_dir
'''
DESCRIPTION = """Exctracts 1 frame per second from video."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('video', help='Path to video.')
    parser.add_argument('target_dir', help='Directory to save frames in.')
    args = parser.parse_args()
    main(args)
