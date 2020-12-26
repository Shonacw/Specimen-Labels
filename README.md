
# Specimen-Label Image processing project in python, Summer 2020. 

## Notes on Labels_Identification_and_Dimensions.py
PREPARE CONTOURS [Function Name = 'Get_All_Contours()']
Step 1: Perform Canny edge detection with set parameters and find contours. \n
Step 2: Perform morphological close operation and find contours again on result. \n
Step 3: Refine contour search and order the results by descending area. \n

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

Single-Hashed lines are optional lines of code
Double-Hashe lines are my notes
-------------------------------------------------------------------------------
## NOTES REFERENCED IN CODE (Labels_Identification_and_Dimensions.py)
Note A: 150, 200 work for pic 94 when resized 
        0, 220 work for pic 256
        69, 100 works for pic 81 when resized 
        either of the last two options work for pic 84 when resized
        
Note B: Here it is easier to just carry area forward, otherwise not consistent 
        - i.e.for circle only carrying forward the radius whereas for rectangle 
        would carry forward width and height.
        
Note C: Here, 'ma' is major axis and 'MA' is minor axis.
Note D: Once we know a certain contour contains the label, we know it cannot 
        also contain the scale bar. We therefore have confidence in removing 
        all further contours in the vicinity of the label from our search.
