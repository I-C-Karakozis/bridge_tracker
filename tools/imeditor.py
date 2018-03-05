import cv2
import numpy as np

import Cards
import stats

### ---- INITIALIZATION ---- ###

# confidence level for background/foreground predictions
CONFIDENCE = 99

# we use 3 degrees of freedom bc we use the RGB color space
DOF = 3

# despeckling configuration
MEDFILT_ITERS = 1
MEDFILT_SIZE = 5

# threshold
WHITE_T = 200

# grayscale label values
WHITE = 255
RED = 60
BLACK = 0

# red and black suit
RED_S = ['H', 'D']
BLACK_S = ['S', 'C']

# color histogram thresholds:
COLOR_RED_T = 100 # red channel only
COLOR_BLACK_T = 80 # all channels

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

    # find foreground mask
    zscore = np.square((img - mu).astype(np.float) / sigmas)
    zscore = np.reshape(np.apply_over_axes(np.sum, zscore, [2]), (dims[0], dims[1]))
    zscore = np.repeat(zscore[:, :, np.newaxis], 3, axis=2)
    mask = np.where(stats.h_test(zscore, confidence=CONFIDENCE, 
                                       dof=DOF), 0, 255).astype(np.uint8)

    # despeckle
    for j in range(MEDFILT_ITERS):
        mask = cv2.medianBlur(mask, MEDFILT_SIZE)

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

def label_pixels(image, suit):
    ''' Label all pixels of image as white and 
        black or red based on its suit.'''

    for pixel in np.nditer(image, op_flags=['readwrite']):
        if pixel >= WHITE_T:
            pixel[...] = WHITE
        elif suit in BLACK_S:
            pixel[...] = BLACK
        elif suit in RED_S:
            pixel[...] = RED
        else:
            print("Bad file name.", img_file)
            exit()

def WRB_histogram(image):
    ''' Compute white-red-black color histogram of image'''
    red = 0
    black = 0
    white = 0

    dims = np.shape(image)
    for row in range(dims[0]):
        for col in range(dims[1]):
            if max(image[row][col]) < COLOR_BLACK_T:
                 black = black + 1
            elif image[row][col][2] - image[row][col][0] > COLOR_RED_T and image[row][col][2] - image[row][col][1] > COLOR_RED_T:
                red = red + 1
            else:
                white = white + 1

    return white, red, black
    