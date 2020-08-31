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
                help = "Path to tangible program image")
ap.add_argument('-d', '--debug', action='store_true',
                help = "basic debug output")
ap.add_argument('-p', '--perspective', action='store_true',
                help = 'perspective transformation debug output') 
ap.add_argument('-a', '--all', action='store_true',
                help = "Activate debug output")
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
low_filter = .0002*np.prod(image.shape) 

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

    block_mask = cv.dilate(block_mask, mask_dilation,iterations = 3)

    # Masking to extract image feature from image
    feature = cv.bitwise_and(gray_A, gray_A, mask=block_mask)  

    # Save block information
    nb = tg.new_block('unprocessed', x, y, w, h, feature)

    sorted_blocks = sorted(tg.block_list, key=lambda block: block.coord_sum)

    #for block in sorted_blocks:
    #    plt.title('{}:{}'.format(block.b_id, block.coord_sum))
    #    plt.imshow(block.feature)
    #    plt.show()

counter = 0
height_sum = 0
for block in sorted_blocks:
    height_sum += block.height
    counter +=1 

mean_height = height_sum/counter

#############################
# 3. Preparing text for OCR #
#############################

print('Found {} blocks in image'.format(counter))
print('OCR')
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
    if block.height > mean_height: # TODO: unreliable
        crop_h = int(.5*block.height)
        feature = block.feature[block.y+crop_v: block.y+block.height-crop_h, 
                                block.x+crop_v: block.x+block.width-crop_right]
    else:
        feature = block.feature[block.y+crop_v:block.y+block.height-crop_v, 
                                block.x+crop_v: block.x+block.width-crop_right]
    
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
    text_in_block = tg.similar_to_exp_text(text_in_block)
    
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

# Root of the AST
root = Node('tangible program')
next_block = None
# current indentation level
nesting_level = root   
# init first node
start = Node(tg.block_list[0].text, parent=nesting_level) 
previous_node = start
  # save all nesting levels so i can go back
nesting_levels = []   
nesting_levels.append(nesting_level)
 # index of nesting_level list
index = 0              

# Assuming all blocks in the list are sorted
for i in range(1, len(sorted_blocks)):

    ## "Special blocks" ##
    if tg.similar(sorted_blocks[i].text, 'b_tab') > 0.7: 
        # TODO: This can be done without relying on this blocks text

        if index != 0:
            index -= 1
        continue

    if tg.similar(sorted_blocks[i].text, 'repeat indefinitely do') > 0.7 : 
        # TODO: or any other control block
        new_node = Node(sorted_blocks[i].text, parent=nesting_levels[index])
        previous_node = new_node
        nesting_level = new_node
        nesting_levels.append(nesting_level)
        index += 1
        continue
    ##  #  #  #  #  #  ##

    res = tg.get_attached_to(sorted_blocks[i-1])
    if res != None and res.b_id == sorted_blocks[i].b_id:
        # the block is to the right of the previous block
        # attach node to previous Node
        new_node = Node(res.text, parent=previous_node)
        previous_node = new_node
        continue

    res = tg.get_indented_to(sorted_blocks[i-1])
    if res != None and res.b_id == sorted_blocks[i].b_id:
        # print('{} is indented to {}'.format(res.text,block_list[i-1].text))
        # the block is indented to the previous block
        #print('{} is indented to {}'.format(sorted_blocks[i], sorted_blocks[i-1]))
        new_node = Node(res.text, parent=previous_node)
        nesting_level = new_node
        nesting_levels.append(nesting_level)
        index += 1
        previous_node = new_node
        continue

    res = tg.get_underneath(sorted_blocks[i-1])
    if res != None and res.b_id == sorted_blocks[i].b_id:
        # the block is underneath the previous block

        #if tg.similar(sorted_blocks[i-1].text, 'Start') > 0.7:
        #    new_node = Node(sorted_blocks[i].text, parent=previous_node)
        #else:
        new_node = Node(sorted_blocks[i].text, parent=nesting_levels[index])
        previous_node = new_node
        continue

    if res == None:
        # Nothing of the previous, add it to the current nest level
        #print('{} got to None FROM {}'.format(sorted_blocks[i], sorted_blocks[i-1]))
        new_node = Node(sorted_blocks[i].text, parent=nesting_levels[index])
        previous_node = new_node
        continue

print('Printing AST...')   
tg.print_AST(root)