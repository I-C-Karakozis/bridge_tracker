import cv2
import numpy as np

import Cards
import stats

from scipy.signal import argrelextrema, argrelmax

### ---- INITIALIZATION ---- ###

# confidence level for background/foreground predictions
CONFIDENCE = 99

# we use 3 degrees of freedom bc we use the RGB color space
DOF = 3

# threshold
WHITE_T = 200

# grayscale label values
WHITE = 255
RED = 60
BLACK = 0

# red and black suit
RED_S = ['H', 'D']
BLACK_S = ['S', 'C']

# color histogram LOW percentile
LOW = 10

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

def orient_card(corner, corner_rotated):
    ''' 
    Identify which corner has the most non-white pixels 
    from the color histogram. Returns 0 for original orientation, 
    1 for rotated. Returns median and LOWth percentile value for each 
    color channel.
    '''
    dims = np.shape(corner)
    pixels = corner.reshape((dims[0]*dims[1], 3))
    pixels_rotated = corner_rotated.reshape((dims[0]*dims[1], 3))

    # find median and low value for each color channel for original orientation
    pixels.sort(axis=0)
    m_half = pixels[dims[0]*dims[1]/2]
    m_low = pixels[dims[0]*dims[1]/10]

    # find median and low value for each color channel for flipped orientation
    pixels_rotated.sort(axis=0)
    m_half_rot = pixels_rotated[dims[0]*dims[1]/2]
    m_low_rot = pixels_rotated[dims[0]*dims[1]/LOW]
    
    # determine orientation by comparing color histograms
    print("Orig:", sum(m_low[0:2]), "- Flipped:", sum(m_low_rot[0:2]))
    print("Red:", m_low[2], "- Red Flipped:", m_low_rot[2])
    if sum(m_low[0:2]) <= sum(m_low_rot[0:2]):
        return 0, m_half, m_low
    else:
        return 1, m_half_rot, m_low_rot

def despeckle(img, thresh):
    """ 
    Remove all connected components below size tresh on binary img.
    """
    # find all connected components; 8-way connectivity
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # take out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 10  
    
    # for every component in the image, you keep it only if it's above min_size
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2
    