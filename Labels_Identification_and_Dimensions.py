#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:07:21 2020

@author: ShonaCW

PREPARE CONTOURS [Function Name = 'Get_All_Contour()s']
Step 1: Perform Canny edge detection with set parameters and find contours.
Step 2: Perform morphological close operation and find contours again on result.
Step 3: Refine contour search and order the results by descending area.


OBTAINING AND RETURNING LABEL INFORMATION [Function Name = 'Info_Shape_Size()']
Step 1: Obtain full list of Contours contained in the image, as described above.
Step 2: One-by-one, outline the contour on the image and ask user whether the
        specimen label has been succesfully identified by the contour.
Step 3: Once the contour containing the specimen label has been identified, 
        remove all other contours contained in that area (refine contour search).
Step 4: Ask user whether there is a scale bar included in the image...
        If NO -> calculate dimensions of the specimen label in units of pixels 
                 and return information to the user.
        If YES-> repeat Step 2 but this time asking whether the scale bar has
                 succesfully identified. Once identified, calculate the dimensions
                 of the label in units of the scale bar and return label 
                 information.
    
-------------------------------------------------------------------------------
MINI READ-ME
# Single-Hashed lines are optional lines of code
## Double-Hashe lines are my notes

-------------------------------------------------------------------------------
NOTES REFERENCED IN CODE
Note A: 150, 200 work for pic 94 when resized 
        0, 220 work for pic 256
        69, 100 works for pic 81 when resized 
        either of the last two options work for pic 84 when resized
        
Note B: Here it is easier to carry area forward than all info bc that is not
        consistent-i.e.for circle only carrying forward the radius whereas
        for rectangle would carry forward width and height.
        
Note C: Here, 'ma' is major axis and 'MA' is minor axis.
"""

#IMPORTING MODULES AND DEFINING REQUIRED FUNCTIONS.
import matplotlib.pyplot as plt
import numpy as np
import cv2

def Get_All_Contours(image):
    ##Option to downscale image, as Canny tends to perform better on smaller images...
    #w, h, c = image.shape
    #resize_coeff = 0.25
    #image = cv2.resize(image, (int(resize_coeff*h), int(resize_coeff*w)))
    
    """Step 1"""
    canny = cv2.Canny(image, 50, 220) ##see Note A 
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    """Step 2"""
    w, h, c = image.shape
    blank = np.zeros((w, h)).astype(np.uint8)
    cv2.drawContours(blank, contours, -1, 1, 1)
    blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    """Step 3"""
    contours = [c for c in contours if 1800 < cv2.contourArea(c)] 
    
    ##If assuming the labels are rectangle, can refine search further with only the 
    ##contours whih are BEST described by a rectange...
    #contours = [c for c in contours if decide_shape_to_plot(c)[0] == 0]
    
    areaArray = []
    for i, a in enumerate(contours):
        area_ = cv2.contourArea(a)
        areaArray.append(area_)
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    return sorteddata

def yes_or_no(question):
    reply = str(input(question +' (yes/no): ')).lower().strip()
    if reply == 'yes':
        return True
    if reply == 'no':
        return False
    
def scale_bar_scale():
    reply = str(input('Is this an NMS Scale bar? (yes/no): ')).lower().strip()
    if reply == 'yes':
        return True
    if reply == 'no':
        rep_2 = str(input('What is the scale? (i.e. cm or mm): ')).lower().strip()
        return rep_2

def plot_rectangle_simple(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 8)
    
    plt.imshow(image)
    return w, h

def area_rectangle(box):
    area = 0
    for i in [0,1,2,3]:
        x = box[i][0]
        y_ = box[i][1]
        
        if i+1 == 4:
            i = -1
        y = box[i+1][1]
        x_ = box[i+1][0]
        area += (x*y - x_*y_)
    area = area * 0.5
    return area

def decide_shape_to_plot(c):
    """
    Function takes a contour c and decides whether its best-approximated by a 
    rectangle, circle, or ellipse. Returns the contour index, shape, area and
    dimensions.
    
    See Note B.
    """
    shape = ['Rectangle', 'Circle', 'Ellipse']
    areas = [] ##List will contain the area of each of the bounding shapes
    diff = [] ##List of the difference in area between contour and bounding shape
    dimensions = [] ##List for the dimensions of the bounding shapes
    
    countour_area = cv2.contourArea(c)
    
    #------------BOUNDING RECTANGLE--------------------------------------------
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    area_rec = area_rectangle(box)

    areas.append(area_rec)
    diff.append(area_rec - countour_area)
    dimensions.append((w, h))
    
    #------------BOUNDING CIRCLE-----------------------------------------------
    contours_poly = cv2.approxPolyDP(c, 3, True)
    center, radius = cv2.minEnclosingCircle(contours_poly)
    area_circ = 3.1415 * (radius**2)

    areas.append(area_circ)
    diff.append(area_circ - countour_area)
    dimensions.append((radius))
    
    #-------------BOUNDING ELLIPSE---------------------------------------------
    (x, y), (MA, ma), angle = cv2.fitEllipse(c)
    area_ell = MA * ma ##Note C
    
    areas.append(area_ell)
    diff.append(area_ell - countour_area)
    dimensions.append((ma, MA))
    
    ##Decide which bounding-shape suits this contour best
    index = np.argmin(diff) 
    return index, areas[index], shape[index], dimensions[index]

def Remove_Section_Inside_Contour(contour, image):
    """
    Function for refining search. Once we know a certain contour contains the 
    label, we know it cannot also contain the scale bar. We therefore have
    confidence in removing all further contours in this area from the search.
    
    Although not the most efficient method, currently masking the area inside 
    the Label contour then performing another search for the contours in the
    image.
    """
    
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    cv2.drawContours(mask, [contour], -1, 0, -1)
    # remove the contours from the image and show the resulting images
    image = cv2.bitwise_and(image, image, mask = mask)
    
    sorteddata = Get_All_Contours(image) #re-define sorteddata on masked image
    return sorteddata

def Check_Contours(image, Contours, Question):
    i = 0
    while True:
        ##Re-define image to avoid plotting over the top of an edited version
        img = image.copy()
        ##Define contour to be investigated
        c = Contours[i][1]
        
        ##Option to rescale image; Canny tends to work better on smaller images
        #w_, h_, c_ = img.shape
        #resize_coeff = 0.25
        #img = cv2.resize(img, (int(resize_coeff*h_), int(resize_coeff*w_)))

        ##Plot rectanglular contour
        w, h = plot_rectangle_simple(c, img)
        plt.pause(0.1)
        
        ##Check if this contour contains the Specimen Label
        answer = yes_or_no(Question)
        if answer == False:
            i += 1          ##if not, we keep looping...
            plt.close()
        if answer == True:
            area = round(w * h, 3)
            dimensions = (w, h)
            
            ##Obtain a new list of contours, ignoring area enclosed by the label
            sorteddata = Remove_Section_Inside_Contour(c, img)
            plt.close()
            return area, dimensions, sorteddata
        
def Info_Shape_Size(img):
    """
    The main function. Deciphers the shape and dimensions of the label.
    """
    sorteddata = Get_All_Contours(img)
    #collect information about the label
    area_L, dimensions_L, sorteddata_mid = Check_Contours(img, sorteddata, "Is this the Label?")

    #if applicable, collect information about the scale bar
    answer = yes_or_no('Is there a scale bar?')
    if answer == False:
        return area_L, dimensions_L
    if answer == True:
        area_S, dimensions_S, sorteddata = Check_Contours(img, sorteddata_mid, "Is this the Scale Bar?")
    
    #Currently assuming the scale bar will be rectangular
    answer = scale_bar_scale()
    if answer == True:
        #calculation for an RMS scale bar
        dim_scale = dimensions_S[0] / 2
        real_dim = [round(dimensions_L[i] / dim_scale, 3) for i in range(len(dimensions_L))]
        scaled_dim = f'Dimensions: {str(real_dim)[1:-1]} [cm]'
        return scaled_dim
        
        
    else:
        #calculation for another tysaape of scale bar
        real_dim = [round(dimensions_L[i] / dimensions_S[0], 3) for i in range(len(dimensions_S))]
        scaled_dim = f'Dimensions: {str(real_dim)[1:-1]} [{answer}]'
        return scaled_dim


#%%
##THE CODE.
img = cv2.imread('9.jpg')
Info_Shape_Size(img)

#NOTE FOR SELF: the masking works but the improved sorteddata isnt working.


#%%
##IGNORE. Extras for exploring the results.
for c in contours:
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,191,255),5)
plt.imshow(img)
 
#%%
plt.close()
