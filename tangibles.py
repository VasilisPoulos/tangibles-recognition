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

# Pre-processing constants
DILATION_WINDOW_SIZE = 3
DILATION_ITERATIONS = 5
EROSION_WINDOW_SIZE = 3
EROSION_ITERATIONS = 3

# Other constants
START_HEIGHT = 0
MASK_WS = 4
mask_dilation = cv.getStructuringElement(cv.MORPH_RECT,\
    (MASK_WS, MASK_WS))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = "path to tangible program image")
ap.add_argument('-d', '--debug', action='store_true',
                help = "basic debug output")
ap.add_argument('-a', '--all', action='store_true',
                help = "all debug output")
args = vars(ap.parse_args())

# Load image
image = cv.imread(args['image'])

#############################################
# 1. Preparing the image for pre-processing.#
##############################################

ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)


# In case automatic Perspective transformation fails the user is prompted to 
# select manually the paper's edges.
# TODO: this is not reliable, the 'experiment' images that are processed 
# and produce correct results use the automatic method.
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
    # if fails make user choose edges manually
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

# Original image to grayscale
gray_A = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Threshold saturation chanel
ret3, th_saturation = cv.threshold(s, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# Dilating thresholded image to remove letters
kernel = cv.getStructuringElement(cv.MORPH_RECT,\
    (DILATION_WINDOW_SIZE,DILATION_WINDOW_SIZE))
th_saturation = cv.dilate(th_saturation, kernel \
                        ,iterations = DILATION_ITERATIONS)

# Erosion to remove noise
kernel = np.ones((EROSION_WINDOW_SIZE,EROSION_WINDOW_SIZE),np.uint8)
th_saturation = cv.erode(th_saturation,kernel,iterations = EROSION_ITERATIONS)

# Using connected components (CC) method to label each block
num_labels, labels_im = cv.connectedComponents(th_saturation)

# Used for filtering components
low_filter = .0005*np.prod(image.shape) 

# Plotting general debug images
if args['debug']:
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(orig, cv.COLOR_BGR2RGB))
    plt.subplot(1, 4, 2)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.subplot(1, 4, 3)
    plt.imshow(th_saturation, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.imshow(th_saturation)
    plt.show()

counter = 0
print('Collecting block information')
for num in range(1, num_labels):

    # Extracting mask from (CC) result
    label = labels_im == num
    block_mask = np.copy(th_saturation)
    block_mask[label == False] = 0
    block_mask[label == True] = 255 
    
    # Find coordinates (x, y) and hight - width of the block (h, w) based on 
    # the blocks mask
    contours, _ = cv.findContours(block_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv.boundingRect(contours[0])

     # Filtering features
    if cv.countNonZero(block_mask) < low_filter:
        continue
    
    #plt.imshow(block_mask)
    #plt.show()

    block_mask = cv.dilate(block_mask, mask_dilation,iterations = 3)

    # Masking to extract image feature from image
    feature = cv.bitwise_and(gray_A, gray_A, mask=block_mask)  

    # Save block information
    nb = tg.new_block('unprocessed', x, y, w, h, feature)

    sorted_blocks = sorted(tg.block_list, key=lambda block: block.coord_sum)

    counter +=1 
    #for block in sorted_blocks:
    #    plt.title('{}:{}'.format(block.b_id, block.coord_sum))
    #    plt.imshow(block.feature)
    #    plt.show()


height_sum = 0
for block in sorted_blocks:
    height_sum += block.height
    

mean_height = height_sum/counter

#############################
# 3. Preparing text for OCR #
#############################

print('Found {} blocks in image'.format(counter)) # TODO: change counter
# Kernels that will be used in erosion-dilation morphological transformations
erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

for block in sorted_blocks:
    # Cropping each block to help with binarization later
    # Using a temporary cropping solution to crop control blocks 
    # TODO: find a better way to crop all blocks 
    # (maybe use image_to_data() from Tesseract)
   
    crop_v = 18
    crop_right = int(.08*block.width)
    if block.height > mean_height or block.height > 150: # TODO: unreliable
        crop_h = int(.58*block.height)
        feature = block.feature[block.y+crop_v: block.y+block.height-crop_h, 
                                block.x+crop_v: block.x+block.width-crop_right]
    else:
        feature = block.feature[block.y+crop_v:block.y+block.height-crop_v, 
                                block.x+crop_v: block.x+block.width-crop_right]
    
    if feature.size == 0: continue
    # Upscale feature image
    multiplier = 2
    feature = cv.resize(feature, dsize=(feature.shape[1]*multiplier, \
        feature.shape[0]*multiplier), interpolation=cv.INTER_CUBIC)

    # blurring to make bin more accurate
    feature = cv.GaussianBlur(feature,(5,5),0)

    if args['debug']:
        plt.imshow(feature, cmap='gray')
        plt.show()

    # Binarize and invert feature
    ret3, th_feature = cv.threshold(feature, 0, 255, \
        cv.THRESH_BINARY+cv.THRESH_OTSU)
    #closing = cv.morphologyEx(th_feature, cv.MORPH_CLOSE, closing_kernel)
    erosion = cv.erode(th_feature, erosion_kernel, iterations = 1)
    dilation = cv.dilate(erosion, dilation_kernel,iterations = 1)
    inv_feature = np.invert(dilation)

    ######################
    # 4. Collecting data #
    ######################
    
    # Tesseract 
    config = '--psm 7'
    tesseract_output = pytesseract.image_to_string(inv_feature, \
            lang='eng', config=config)

    # Remove spaces and newlines
    text_in_block = ' '.join(tesseract_output.split())
    # Match text to expected text
    if not (tg.similar(text_in_block, 'Variable') > 0.8): 
        text_in_block = tg.similar_to_exp_text(text_in_block)
        block.set_text(text_in_block)
    else:
        block.set_text(text_in_block)

    # Print pre-process/ Tesseract results
    if args['debug'] and args['all']:
        plt.title('Block\'s id: {} at ({}, {})\n Tesseract read: {}'\
            .format(nb.b_id, nb.x, nb.y, ' '.join(tesseract_output.split())))
        plt.imshow(inv_feature, cmap='gray')
        plt.show()

#####################
# 5. AST generation #
#####################

# Base anytree library node, used to connect all nodes of the tangible program
root = Node('tangible program')
previous_node = root

block_has_attached = False
nesting_block_list = [] # Code blocks that are before each 
                        # nested coding block set.
nesting_node_list = []

LIST_INDEX = 0

handling_control_block = False

# add node based on nesting level
def add_node_on_nest_lvl(block):
    if len(nesting_node_list) == 0:
        #print('> added node {} to {}'.format(current_block.text, root))
        previous_node = Node(block.text, parent=root)
    else:
        #print('> added node {} to {}'.format(current_block.text, nesting_node_list[-1]))
        previous_node = Node(block.text, parent=nesting_node_list[-1])

    return previous_node

def get_next_block():
    # Simple function to improve redability below
    # returns the next block in sorted_block list.
    global LIST_INDEX
    if LIST_INDEX >= len(sorted_blocks):
        return None
    ret = sorted_blocks[LIST_INDEX]
    LIST_INDEX += 1
    return ret


current_block = get_next_block()
handle_control_indented = False
if_do_ind_block = None
while current_block != None:
    #print('processing {} ... \t\t LEVEL {}'.format(current_block.text,  len(nesting_node_list)))

    if tg.similar(current_block.text, 'unprocessed') > 0.7:
        current_block = get_next_block()
        continue

    if tg.similar(current_block.text, 'b_tab') > 0.7:
        nesting_node_list.pop()
        #print('> nesting changed to LEVEL {}'.format(len(nesting_node_list)))
        current_block = get_next_block()
        continue
        
    # Case 1 : Blocks attached to the right of current_block.
    attached_to_block_list = []
    next_block = tg.get_block_attached_to(current_block)
    if next_block != None : 
        previous_node = add_node_on_nest_lvl(current_block)

        # Keeping parent information 
        parent_node = previous_node
        parent_block = current_block    # So that we can return to parent_block
                                        # once we found all the attached blocks.
    
        #print('>>>> NEXT_BLOCK: {}'.format(next_block.text))
        # Iterate through attached blocks.
        while next_block != None:
            block_has_attached = True
            #print('>>>> append: {}'.format(next_block.text))
            attached_to_block_list.append(next_block)
            current_block = get_next_block()
            next_block = tg.get_block_attached_to(current_block)

        #print('>>>> ATTACHED: {}'.format(block_has_attached))
        # Printing results.
        if block_has_attached:
            list_str = ''
            for block in attached_to_block_list:
                list_str += str(block.text) + '| '
                #print('> added node {} to {}'.format(block.text, previous_node))
                previous_node = Node(block.text, parent=previous_node)
                
            block_has_attached = False
            current_block = parent_block    # using the parent_block variable 
                                            # to continue searching
            previous_node = parent_node     # reset tree node to parent node
            #print('CASE 1: {} has {} attached'.format(parent_block.text, list_str))
            current_block = get_next_block()

            if tg.is_control_block(parent_block.text):      # if do
                if_do_ind_block = tg.get_block_indented_to(parent_block)
                if current_block.b_id == if_do_ind_block.b_id:
                    #print('NAI RE EINAI '+ str(nesting_node_list))
                    nesting_node_list.append(parent_node)
                    handle_control_indented = True
                    continue
            else:
                continue
    
    # Case 2
    next_block = tg.get_block_indented_to(current_block)
    if next_block != None:
        #print('CASE 2: {} has {} indented'.format(current_block.text, next_block.text))

        # add current block to tree (current indentation lvl)
        previous_node = add_node_on_nest_lvl(current_block)
        # append the current node to the indentation list
        #print('>>>> ADD TO NESTING {}'.format(previous_node))
        nesting_node_list.append(previous_node) 
        # print the change
        #print('> nesting changed to LEVEL {}'.format(len(nesting_node_list)))

        current_block = get_next_block() #this is the actual indented node
        handling_control_block = False
        continue

    # Case 3
    next_block = tg.get_block_underneath(current_block)
    if next_block != None:
        #print('CASE 3: {} has {} underneath'.format(current_block.text, next_block.text))
        previous_node = add_node_on_nest_lvl(current_block)
        current_block = get_next_block()
        continue
    
print('Printing AST...')   
tg.print_AST(root)