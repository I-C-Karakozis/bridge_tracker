import argparse
import cv2
import numpy as np

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

def main(args):

    img = cv2.imread(args.test_img)
    green_mu, green_sigmas = model_boundary(img)
    no_felt = remove_background(img, green_mu, green_sigmas)

    cv2.imshow("no_felt", no_felt)
    key = cv2.waitKey(100000) & 0xFF
    if key == ord("q"):
        return

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
