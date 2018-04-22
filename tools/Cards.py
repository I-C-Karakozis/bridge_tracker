############## Playing Card Detector Functions ###############
#
# Code borrowed by Evan Juras (with minor modifications and additions)
# Source: https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm

from sklearn.cluster import KMeans
import numpy as np

import cv2
import imutils
import sys
import time

from tools import imeditor

### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60

CARD_MAX_AREA = 1200000
CARD_MIN_AREA = 1000 #25000

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 26
CORNER_HEIGHT = 50

# Number of pixels to curtail from the edges to ensure no background is getting caught
CURTAIL_H = 8
CURTAIL_W = 4

SCALE = 1.0 / 1.5

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.color_warp = [] # 200x300, flattened, color, blurred image
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.best_match = "Unknown" # Best matched rank
        self.diff = 0 # Difference between card image and best matched gt image

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    retval, thresh = cv2.threshold(blur,BKG_THRESH,255,cv2.THRESH_BINARY)
    
    return thresh

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp, qCard.color_warp = flattener(image, pts, w, h)

    return qCard

def extract_card(contour, image):
    """Uses contour to find extract warped card image."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour
    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp, qCard.color_warp = flattener(image, pts, w, h)

    return qCard.color_warp

def template_match(warp, suit, train_images, train_labels):
    '''
    Label all pixels of the warp based on the specified suit.
    Using the distance transform, perform template matching against train images with groundtruth train_labels.
    Return best match.
    '''
    best_match_diff = sys.maxint
    best_match = "Unknown" 

    # label pixels and despeckle
    imeditor.label_pixels(warp, suit[0])
    imeditor.despeckle(warp, np.max(warp))
    warp = cv2.bitwise_not(warp)

    # threshold cards to separate foreground elements from white background elements                
    warp_cur = warp[CURTAIL_H:-CURTAIL_H, CURTAIL_W:-CURTAIL_W]
    ret, orig = cv2.threshold(warp_cur, imeditor.RED+1, 255, cv2.THRESH_BINARY)

    # template matching
    for gt_img, gt_label in zip(train_images, train_labels):

            if gt_label[-1] in suit:
                # Curtail boundaries of card for to remove background that might have been included
                # in the crop and can ruin the template matching
                gt_img_cur = gt_img[CURTAIL_H:-CURTAIL_H, CURTAIL_W:-CURTAIL_W]

                # threshold cards to separate foreground elements from white background elements                
                ret, gt = cv2.threshold(gt_img_cur, imeditor.RED+1, 255, cv2.THRESH_BINARY)

                # perform 2-way distance transform
                cv2.distanceTransform(orig, maskSize=cv2.DIST_MASK_PRECISE, distanceType=cv2.DIST_L2)
                diff = np.sum(cv2.distanceTransform(orig, maskSize=cv2.DIST_MASK_PRECISE, distanceType=cv2.DIST_L2) * np.invert(gt)) + np.sum(np.invert(orig) * cv2.distanceTransform(gt, maskSize=cv2.DIST_MASK_PRECISE, distanceType=cv2.DIST_L2))
                if diff < best_match_diff:
                    best_match = gt_label
                    best_match_diff = diff

    return best_match, best_match_diff
  
def match_card(qCard, train_labels, train_images):
    """Finds best card match for the query card. Differences
    the query card image with the groundtruth card images.
    The best match is the groundtrugh that has the least difference."""  
    
    # obtain 90-rotated color warp
    dst = imutils.rotate_bound(qCard.color_warp, 90)
    rotated_color_warp = cv2.resize(dst, (200, 300))  
    rotated_warp = cv2.cvtColor(rotated_color_warp, cv2.COLOR_BGR2GRAY) 

    # obtain corners
    Qcorner = qCard.color_warp[CURTAIL_H:CORNER_HEIGHT, CURTAIL_W:CORNER_WIDTH]
    Qcorner_rotated = rotated_color_warp[CURTAIL_H:CORNER_HEIGHT, CURTAIL_W:CORNER_WIDTH]

    # obtain correct card orientation from color histogram
    orientation, half, low = imeditor.orient_card(Qcorner, Qcorner_rotated)

    if orientation == 0:

        # identify card color from color histogram of the card corner    
        red = float(low[2]) / half[2]
        other =  float(low[0]) / half[0] + float(low[1]) / half[1]
        if red >= SCALE*other:
            suit = imeditor.RED_S
        else:
            suit = imeditor.BLACK_S 

        best_match, best_match_diff = template_match(qCard.warp, suit, train_images, train_labels)

    else:

        # identify card color from color histogram of the card corner  
        red = float(low[2]) / half[2]
        other = float(low[0]) / half[0] + float(low[1]) / half[1]
        if red >= SCALE*other:
            rotated_suit = imeditor.RED_S
        else:
            rotated_suit = imeditor.BLACK_S 

        best_match, best_match_diff = template_match(rotated_warp, rotated_suit, train_images, train_labels)

    return best_match, best_match_diff
    
def draw_results(image, qCard):
    """Draw the card label, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    label = qCard.best_match

    # Draw card label twice, so letters have black outline
    cv2.putText(image,(label),(x-60,y-10),font,1.5,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(label),(x-60,y-10),font,1.5,(0,0,255),2,cv2.LINE_AA)

    return image, label

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    color_warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    gray_warp = cv2.cvtColor(color_warp,cv2.COLOR_BGR2GRAY)        

    return gray_warp, color_warp
