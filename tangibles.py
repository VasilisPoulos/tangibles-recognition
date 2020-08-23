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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = "Path to tangible program image")
ap.add_argument('-d', '--debug', action='store_true',
                help = "Activate debug output")
ap.add_argument('-a', '--all', action='store_true',
                help = "Activate debug output")
args = vars(ap.parse_args())

# Open image
image = cv.imread(args['image'])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# Perspective transformation
screenCnt = tg.find_points(image)
image = tg.four_point_transform(orig, screenCnt.reshape(4,2)*ratio)

# Original image to HSV
hsv_A = cv.cvtColor(image, cv.COLOR_RGB2HSV)

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

# Erosion 
kernel = np.ones((EROSION_WINDOW_SIZE,EROSION_WINDOW_SIZE),np.uint8)
th_saturation = cv.erode(th_saturation,kernel,iterations = EROSION_ITERATIONS)

# Using connected components (CC) method to label each block
num_labels, labels_im = cv.connectedComponents(th_saturation)

# Kernels that will be used in erosion-dilation morphological transformations
erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

# Used for filtering components
low_filter = .0002*np.prod(image.shape) 

if args['debug']:
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(orig, cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.imshow(th_saturation)
    plt.show()

block_counter = 0
for num in range(1, num_labels):

    # Extracting mask from (CC) result
    label = labels_im == num
    block_mask = np.copy(th_saturation)
    block_mask[label == False] = 0
    block_mask[label == True] = 255


    # Find coordinates (x, y) and hight - width of the block (h, w) based on 
    # the blocks mask
    _, contours, _ = cv.findContours(block_mask, cv.RETR_TREE, \
            cv.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv.boundingRect(contours[0])

    
    # Masking to extract image feature from image
    feature = cv.bitwise_and(gray_A, gray_A, mask=block_mask)  
    
    if cv.countNonZero(feature) < low_filter:
        continue

    # Cropping each block to help with binarization later
    # Using a temporary cropping solution to crop control blocks 
    # TODO: find a better way to crop all blocks 
    # (maybe use image_to_data() from Tesseract)
    crop_v = 18
    crop_right = int(.08*w)
    if h > 150:
        crop_h = int(.5*h)
        print(h, crop_h)
        feature = feature[y+crop_v:y+h-crop_h, x+crop_v: x+w-crop_right]
    else:
        feature = feature[y+crop_v:y+h-crop_v, x+crop_v: x+w-crop_right]
    
    # Upscale feature image
    multiplier = 2
    feature = cv.resize(feature, dsize=(feature.shape[1]*multiplier, \
        feature.shape[0]*multiplier), interpolation=cv.INTER_CUBIC)

    # Binarize and invert feature
    ret3, th_feature = cv.threshold(feature, 0, 255, \
        cv.THRESH_BINARY+cv.THRESH_OTSU)
    #closing = cv.morphologyEx(th_feature, cv.MORPH_CLOSE, closing_kernel)
    erosion = cv.erode(th_feature, erosion_kernel, iterations = 1)
    dilation = cv.dilate(erosion, dilation_kernel,iterations = 1)
    inv_feature = np.invert(dilation)
    
    # Tesseract 
    config = '--psm 7'
    text_in_block = pytesseract.image_to_string(inv_feature, \
            lang='eng', config=config)
   
    # Remove spaces and newlines
    text_in_block = ' '.join(text_in_block.split())

    # Match text to expected text
    text_in_block = tg.similar_to_exp_text(text_in_block)

    # Save block information
    nb = tg.new_block( text_in_block, x, y, w, h)
    
    block_counter += 1
    # Print pre-process/ Tesseract results
    if args['debug'] and args['all']:
        plt.title('Block\'s id: {} at ({}, {})\n Tesseract read: {}'\
            .format(nb.b_id, nb.x, nb.y, ' '.join(text_in_block.split())))
        plt.imshow(inv_feature, cmap='gray')
        plt.show()

print('Found {} blocks in image'.format(block_counter))

## Main logic for generating AST and result ##

# Root of the AST
root = Node('tangible program')
next_block = None
 # current identation level
nesting_level = root   
# init first node
start = Node(tg.get_block_list_item(0).text, parent=nesting_level) 
previous_node = start
  # save all nesting levels so i can go back
nesting_levels = []   
nesting_levels.append(nesting_level)
 # index of nesting_level list
index = 0              

# Assuming all blocks in the list are sorted
for i in range(1, len(tg.block_list)):

    ## "Special blocks" ##
    if tg.similar(tg.block_list[i].text, 'b_tab') > 0.7: 
        # TODO: This can be done without relying on this blocks text
        # print('{} is {} similar to b_tab'.format(block_list[i].text,\
        #   similar(block_list[i].text, 'b_tab')))
        if index != 0:
            index -= 1
        continue

    if tg.similar(tg.block_list[i].text, 'repeat indefinitely do') > 0.7: 
        # TODO: or any other control block
        new_node = Node(tg.block_list[i].text, parent=nesting_levels[index])
        previous_node = new_node
        nesting_level = new_node
        nesting_levels.append(nesting_level)
        index += 1
        continue
    ##  #  #  #  #  #  ##

    res = tg.get_attached_to(tg.block_list[i-1])
    if res != None and res.b_id == tg.block_list[i].b_id:
        # the block is to the right of the previous block
        # attach node to previous Node
        new_node = Node(res.text, parent=previous_node)
        previous_node = new_node
        continue

    res = tg.get_indented_to(tg.block_list[i-1])
    if res != None and res.b_id == tg.block_list[i].b_id:
        # print('{} is indented to {}'.format(res.text,block_list[i-1].text))
        # the block is indented to the previous block
        new_node = Node(res.text, parent=previous_node)
        nesting_level = new_node
        nesting_levels.append(nesting_level)
        index += 1
        previous_node = new_node
        continue

    res = tg.get_underneath(tg.block_list[i-1])
    if res != None and res.b_id == tg.block_list[i].b_id:
        # the block is underneath the previous block

        #if tg.similar(tg.block_list[i-1].text, 'Start') > 0.7:
        #    new_node = Node(tg.block_list[i].text, parent=previous_node)
        #else:
        new_node = Node(tg.block_list[i].text, parent=nesting_levels[index])
        previous_node = new_node
        continue

    if res == None:
        # Nothing of the previous, add it to the current nest level
        new_node = Node(tg.block_list[i].text, parent=nesting_levels[index])
        previous_node = new_node
        continue
   
tg.print_AST(root)