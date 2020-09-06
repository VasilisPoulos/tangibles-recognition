import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv
import pytesseract
import os
import imutils
import argparse
import tang_module as tg
from anytree import Node, RenderTree
from PIL import Image

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = "path to tangible program image")
ap.add_argument('-p', '--pre', action='store_true',
                help = "preprocessing and masks generation debug plots")
ap.add_argument('-o', '--ocr', action='store_true',
                help = "ocr debug plots")
args = vars(ap.parse_args())

# Load image.
image = cv.imread(args['image'])
if image is None or image.size == 0:
    print('the image cannot be read (because of missing file, \
improper permissions, unsupported or invalid format)')
    exit(0)

#############################################
# 1. Preparing the image for pre-processing.#
##############################################

# Resizing image to make processing faster
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# In case automatic Perspective transformation fails the user is 
# prompted to select manually the paper's edges.
# TODO: this doesn't account for user error.
selection_counter = 0
manual_selection = []

def coordinates(event, x, y, flags, param):
    global selection_counter, screenCnt
    
    if event == cv.EVENT_LBUTTONDOWN and selection_counter < 4:
        cv.circle(image, (x,y), 20, (255,200,0), -1)
        # print(x, y)
        manual_selection.append([[x, y]])
        selection_counter += 1
                     
# Automatic point selection; assumes that the hole A4 paper is 
# visible.
screenCnt = tg.find_points(image)

# Manual point selection.
if len(screenCnt) == 0:
    print('Perspective transformation failed')
    cv.namedWindow('point_selection', cv.WINDOW_NORMAL)
    cv.setMouseCallback('point_selection', coordinates)
    print('Choose paper\'s edges manually\npress \'q\' to exit.')
    while(True):
        cv.imshow('point_selection', image)
        key = cv.waitKey(1) & 0xFF
        
        if key == ord("q"):
            cv.destroyAllWindows()
            break
    screenCnt = np.array(manual_selection)
    if selection_counter < 4:
        print('You have to select 4 edges')
        exit(0)

# Perspective transformation
image = tg.four_point_transform(orig, screenCnt.reshape(4,2)*ratio)

################################################
# 2. Pre-Processing and block masks generation.#
################################################

# Color balance 
balanced_img = tg.white_balance(image)

# Original image to HSV
hsv_A = cv.cvtColor(balanced_img, cv.COLOR_RGB2HSV)

# Split each chanel
h,s,v = cv.split(hsv_A)

# Threshold saturation chanel
ret3, th_saturation = cv.threshold(s, 0, 255,\
     cv.THRESH_BINARY+cv.THRESH_OTSU)

# Dilating thresholded image to remove letters
DILATION_WINDOW_SIZE = 3
DILATION_ITERATIONS = 5
EROSION_WINDOW_SIZE = 3
EROSION_ITERATIONS = 3

kernel = cv.getStructuringElement(cv.MORPH_RECT,\
    (DILATION_WINDOW_SIZE,DILATION_WINDOW_SIZE))
th_saturation = cv.dilate(th_saturation, kernel \
                        ,iterations = DILATION_ITERATIONS)

# Erosion to remove noise
kernel = np.ones((EROSION_WINDOW_SIZE,EROSION_WINDOW_SIZE),np.uint8)
th_saturation = cv.erode(th_saturation,kernel,\
    iterations = EROSION_ITERATIONS)

# Using connected components (CC) method to label each block
num_labels, labels_im = cv.connectedComponents(th_saturation)

# Plotting general debug images
if args['pre']:
    fig = plt.figure(figsize=(14,5))
    fig.suptitle('Pre-Processing and block masks generation', \
        fontsize= 18)
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(cv.cvtColor(orig, cv.COLOR_BGR2RGB))
    plt.subplot(1, 4, 2)
    plt.title('Perspective Transformation &\nColor Balance')
    plt.imshow(cv.cvtColor(balanced_img, cv.COLOR_BGR2RGB))
    plt.subplot(1, 4, 3)
    plt.title('Saturation Chanel')
    plt.imshow(s)
    plt.subplot(1, 4, 4)
    plt.title('Thresholded Saturation Chanel')
    plt.imshow(th_saturation, cmap='gray')
    plt.show()

# Original image to grayscale
gray_A = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Used for filtering components
low_filter = .0005*np.prod(image.shape) 

# Dilation to remove letters from masks.
MASK_WS = 4
mask_dilation = cv.getStructuringElement(cv.MORPH_RECT,\
    (MASK_WS, MASK_WS))

counter = 0
print('Collecting block information')
for num in range(1, num_labels):

    # Extracting mask from (CC) result
    label = labels_im == num
    block_mask = np.copy(th_saturation)
    block_mask[label == False] = 0
    block_mask[label == True] = 255 
    
    # Find coordinates (x, y) and hight - width of the block (h, w) 
    # based on the blocks mask
    contours, _ = cv.findContours(block_mask, cv.RETR_TREE, \
        cv.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv.boundingRect(contours[0])

     # Filtering features.
    if cv.countNonZero(block_mask) < low_filter:
        continue
    
    # This is the second time this is applied, this time in each 
    # individual mask.
    block_mask = cv.dilate(block_mask, mask_dilation,iterations = 3)
    
    # Masking to extract image feature from image.
    feature = cv.bitwise_and(gray_A, gray_A, mask=block_mask)  

    # Save block information.
    nb = tg.new_block('unprocessed', x, y, w, h, feature)
    counter +=1 

print('Found {} blocks in image'.format(counter))
sorted_blocks = sorted(tg.block_list, key=lambda block: block.coord_sum)

# Calculating mean height of blocks to be used to identify control
# blocks and correctly crop them out of the original image.
height_sum = 0
for block in sorted_blocks:
    height_sum += block.height
    
mean_height = height_sum/counter

#############################
# 3. Preparing text for OCR #
#############################

# Kernels that will be used in erosion-dilation morphological 
# transformations.
erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

for block in sorted_blocks:
    # Cropping each block to help with binarization later
    # Using a temporary cropping solution to crop control blocks. 

    # TODO: find a better way to crop text of blocks 
    # (maybe use image_to_data() from Tesseract).
   
    crop_v = 18
    crop_right = int(.08*block.width)
    if block.height > mean_height or block.height > 150:
        crop_h = int(.58*block.height)
        feature = \
            block.feature[block.y+crop_v: block.y+block.height-crop_h, 
                        block.x+crop_v: block.x+block.width-crop_right]
    else:
        feature = \
            block.feature[block.y+crop_v:block.y+block.height-crop_v, 
                        block.x+crop_v: block.x+block.width-crop_right]
    
    # Filtering out empty features.
    if feature.size == 0: continue

    # Upscale feature image to have more pixels to wor with 
    # morphological transformations.
    multiplier = 2
    feature = cv.resize(feature, dsize=(feature.shape[1]*multiplier, \
        feature.shape[0]*multiplier), interpolation=cv.INTER_CUBIC)

    # Blurring to make bin more accurate.
    feature = cv.GaussianBlur(feature,(5,5),0)

    # Binarize and invert feature.
    ret3, th_feature = cv.threshold(feature, 0, 255, \
        cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Morphological transformations.
    erosion = cv.erode(th_feature, erosion_kernel, iterations = 1)
    dilation = cv.dilate(erosion, dilation_kernel,iterations = 1)
    inv_feature = np.invert(dilation)

    ######################
    # 4. Collecting data #
    ######################
    
    # Tesseract.
    config = '--psm 7'
    tesseract_output = pytesseract.image_to_string(inv_feature, \
            lang='eng', config=config)

    # Remove spaces and newlines.
    text_in_block = ' '.join(tesseract_output.split())

    # Match text to expected text.
    if not (tg.similar(text_in_block, 'Variable') > 0.8): 
        text_in_block = tg.similar_to_exp_text(text_in_block)
        block.set_text(text_in_block)
    else:
        block.set_text(text_in_block)

    # Print masking - Tesseract results.
    if args['ocr']:
        fig = plt.figure(figsize=(8,5))
        fig.suptitle('Tesseract results & Block Mask', fontsize=18)
        plt.title('Tesseract & Masking ')
        plt.subplot(1,2,1)
        plt.title('Block\'s id: {} at ({}, {})\n Tesseract read: {}'\
            .format(block.b_id, block.x, block.y,\
                 ' '.join(tesseract_output.split())))
        plt.imshow(inv_feature, cmap='gray')
        plt.subplot(1,2,2)
        plt.title('Block\'s Mask')
        plt.imshow(block.feature)
        plt.show()

#####################
# 5. AST generation #
#####################

# Base anytree library node, used to connect all nodes of the tangible
# program.
root = Node('tangible program')
previous_node = root    

# Keepig all the blocks that initiate a new nesting level in this list.
nesting_node_list = []
NESTING_LIST_INDEX = 0

# Add node on the correct nesting level.
def add_node_on_nest_lvl(block):
    if len(nesting_node_list) == 0:
        previous_node = Node(block.text, parent=root)
    else:
        previous_node = Node(block.text, parent=nesting_node_list[-1])
    return previous_node


def get_next_block():
    # Simple function to improve redability below
    # returns the next block in sorted_block list.
    global NESTING_LIST_INDEX
    if NESTING_LIST_INDEX >= len(sorted_blocks):
        return None
    ret = sorted_blocks[NESTING_LIST_INDEX]
    NESTING_LIST_INDEX += 1
    return ret


block_has_attached = False   
# Pointer to correctly connect control blocks that have both attached
# and blocks underneath them (if-do type of blocks).
if_do_ind_block = None
current_block = get_next_block()
while current_block != None:

    # Skip all 'unprocessed' blocks, assuming they will be noise.
    if tg.similar(current_block.text, 'unprocessed') > 0.7:
        current_block = get_next_block()
        continue
    
    # 'b_tab' block removes an indentation level.
    if tg.similar(current_block.text, 'b_tab') > 0.7:
        nesting_node_list.pop()
        current_block = get_next_block()
        continue
        
    # Case 1 : Blocks attached to the right of current_block.
    next_block = tg.get_block_attached_to(current_block)
    if next_block != None : 
        previous_node = add_node_on_nest_lvl(current_block)
        # Keeping parent information so that we can return to 
        # the parent_block once we found all the attached blocks.
        parent_node = previous_node
        parent_block = current_block 
    
        # Iterate through attached blocks.
        while next_block != None:
            block_has_attached = True
            previous_node = Node(next_block.text, parent=previous_node)
            current_block = get_next_block()
            next_block = tg.get_block_attached_to(current_block)

        # Printing results.
        if block_has_attached:
            block_has_attached = False  
            # Using the parent_block variable 
            # to continue searching.
            current_block = parent_block 
            # Reset tree node to parent node.
            previous_node = parent_node    
            current_block = get_next_block()

            if tg.is_control_block(parent_block.text):      
                # In the current state of the project this is reached
                # by if do.
                if_do_ind_block = tg.get_block_indented_to(parent_block)
                if current_block.b_id == if_do_ind_block.b_id:
                    nesting_node_list.append(parent_node)
                    continue
            else:
                continue
    
    # Case 2: block has other blocks indented to it.
    next_block = tg.get_block_indented_to(current_block)
    if next_block != None:
        # Add current block to tree (current indentation lvl).
        previous_node = add_node_on_nest_lvl(current_block)
        # Append the current node to the indentation list.
        nesting_node_list.append(previous_node) 
        # This is the actual indented node.
        current_block = get_next_block() 
        continue

    # Case 3: block has other blocks underneath. 
    next_block = tg.get_block_underneath(current_block)
    if next_block != None:
        previous_node = add_node_on_nest_lvl(current_block)
        current_block = get_next_block()
        continue
    
print('Printing AST...')   
tg.print_AST(root)