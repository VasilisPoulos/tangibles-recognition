from difflib import SequenceMatcher
from anytree import Node, RenderTree
import numpy as np 
import cv2 as cv
import imutils
import matplotlib.pyplot as plt 

EXPECTED_TEXT = ['Start', 
                 'Define',
                 'Variable X',
                 'Variable Y',
                 'Variable Z',
                 '50', '20', '30',
                 'b_tab', 
                 'drive', 'turn', 'forwards speed %',
                 'backwards speed %',
                 'right speed %',
                 'left speed %', 
                 'repeat indefinitely',
                 'if', 'else',
                 'equals', 
                 'get distance US sensor',
                 'right speed %']

CONTROL_BLOCKS = ['repeat indefinitely', 'if', 'else']
# First block id + 1
BLOCK_ID = 0 

# List that holds block objects
block_list = []

class code_block:
    def __init__(self, b_id, text, x, y, w, h, img):
        self.b_id = b_id
        self.text = text
        self.x = x
        self.y = y
        self.width = w
        self. height = h
        self.coord_sum = (x*0.5)+(y*10)
        self.feature = img

    def __str__(self):
        return '{self.b_id}, "{self.text}" at ({self.x}, {self.y}) {self.coord_sum}'.format(self = self)

    def set_text(self, text):
        self.text = text

    def get_coord_sum(self):
        return self.coord_sum

        
def new_block(text, x, y, w, h, img):
    global BLOCK_ID 
    BLOCK_ID += 1
    block_list.append(code_block(BLOCK_ID, text, x, y, w, h, img))
    return block_list[BLOCK_ID - 1]

def find_block(text):
    for block in block_list:
        if block.text == text:
            return block


def get_node_underneath(my_block):
    thr = 30
    res_block =  None
    for block in block_list:
        # Skip the same block as my_block in the block_list or blocks that are 
        # higher in the picture than my_block.
        if block.b_id == my_block.b_id or block.y < my_block.y: 
            # print('Skipped {}'.format(block.text))
            continue
        # Searching for a block that: 
        # * is `not` more to the right or left than my block in the x axis 
        #   + threshold
        # * is `not` lower that my_block's height + threshold
        if  my_block.x - thr < block.x < my_block.x + thr and             block.y < my_block.y + my_block.height + thr:
            res_block = block
    return res_block 


# get_attached_to: returns the attached block to my_block if there is one
# else None.
def get_attached_node_to(my_block):
    thr = 30
    res_block = None
    for block in block_list:
        # Skip the same block as my_block in the block_list or blocks that are 
        # more to the left of the picture than my_block.
        if block.b_id == my_block.b_id or block.x < my_block.x:      
            # print('Skipped {}'.format(block.text))
            continue 
        # Searching for a block that: 
        # * is `not` higher or lower than my block in the y axis + threshold
        # * is `not` more to the right that my_block's width + threshold
        if  my_block.y - thr < block.y < my_block.y + thr and block.x < my_block.x + my_block.width + thr :
            res_block = block
    return res_block 


# get_indented_to: returns the indented block to my_block if there is one
# else None.
def get_indented_node_to(my_block):
    res_block =  None
    thr = 10
    for block in block_list:
        # Skip the same block as my_block in the block_list or blocks that are 
        # higher than my_block.
        if block.b_id == my_block.b_id or block.y < my_block.y: 
            #print('Skipped {}'.format(block.text))
            continue
        # Searching for a block that: 
        # * is within the range of my_block's width + threshold
        # * is below my_block
        if  my_block.x < block.x < my_block.x + my_block.width and  block.y < my_block.y + my_block.height + thr:
            res_block = block
            return res_block


# similar(): 
# returns: a precentage of string a, b similarity
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def similar_to_exp_text(text):
    for line in EXPECTED_TEXT:
        if similar(text, line) > 0.65:
            # print('{} mached with {}'.format(text, line))
            return line
    return text


def is_control_block(text):
    for block_text in CONTROL_BLOCKS:
        if similar(text, block_text) > 0.65:
            # print('{} mached with {}'.format(text, line))
            return True
    return False


def print_AST(root):
    # Print resulting AST
    for pre, _, node in RenderTree(root):
        print("%s%s" % (pre, node.name))

# Perspective transform functions @pyimagesearch.com
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def find_points(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    edged = cv.Canny(gray, 75, 200)
    screen_contours = []
    plt.imshow(edged)
    plt.show()
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_contours = approx
            break

    return screen_contours

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result


