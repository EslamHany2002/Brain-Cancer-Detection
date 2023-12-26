#---------------------------------------------------------- Libraries ---------------------------------------------------------------------
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import skimage.exposure as exposure

#----------------------------------------------- Print Before & After functions -----------------------------------------------------------
def img_diff(Before, After):
    # Determine height and width of images
    h1, w1, channels1 = Before.shape if len(Before.shape) == 3 else (*Before.shape, 1)
    h2, w2, channels2 = After.shape if len(After.shape) == 3 else (*After.shape, 1)

    # Determine the maximum height
    max_height = max(h1, h2)

    # Resize images to have the same height
    Before = cv2.resize(Before, (int(w1 * max_height / h1), max_height))
    After = cv2.resize(After, (int(w2 * max_height / h2), max_height))

    # Create a blank canvas with the maximum height and the sum of the widths
    canvas = np.zeros((max_height, Before.shape[1] + After.shape[1], 3), dtype=np.uint8)

    # Convert images to 3 channels if they have different channels
    if channels1 != 3:
        Before = cv2.cvtColor(Before, cv2.COLOR_GRAY2BGR)
    if channels2 != 3:
        After = cv2.cvtColor(After, cv2.COLOR_GRAY2BGR)

    # Put the Before image on the left side of the canvas
    canvas[:max_height, :Before.shape[1]] = Before

    # Put the After image on the right side of the canvas
    canvas[:max_height, Before.shape[1]:] = After
    
    # Display the result
    window_name = 'Image Processing'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)
    cv2.imshow('Image Processing', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_plot(Before, After):
    if len(Before.shape) == 2:
        # Grayscale image, convert to RGB
        Before = cv2.cvtColor(Before, cv2.COLOR_GRAY2RGB)
    if len(After.shape) == 2:
        # Single-channel image, convert to RGB
        After = cv2.cvtColor(After, cv2.COLOR_GRAY2RGB)
        
    # Plot the images side by side
    plt.subplot(1, 2, 1)
    plt.imshow(Before)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(After)
    plt.title('After Processing')
    
    # Show the plot
    plt.show()
        
#------------------------------------------------- Image TO Grayscale function ------------------------------------------------------------
def cnvrt(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#-------------------------------------------------------- crop function -------------------------------------------------------------------
def crop(path):
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path
    
    # Convert the image to grayscale
    if len(img.shape) > 2:
        img = cnvrt(img)
    gray = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    return new_image

#---------------------------------------------------- Segmentation functions --------------------------------------------------------------
def segment1(path):
    # Image Read
    if isinstance(path, str):
        # Image Read as Grayscale
        image = cv2.imread(path)
    else:
        image = path

    if len(image.shape) == 2:
        # Grayscale image
        grayscale_image = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Color image with three channels
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Handle other cases
        grayscale_image = cnvrt(image)    
        
    # Apply a threshold to the image.
    thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

    # Find the contours in the thresholded image.
    contours = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Draw the contours on the original image.
    segmented_image = image.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 0, 255), 2)

    return segmented_image

def segment2(path):
    # Image Read
    if isinstance(path, str):
        # Image Read as Grayscale
        grayscale_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        grayscale_image = path
        if len(path.shape) == 2:
            # Grayscale image
            grayscale_image = path
        elif len(path.shape) == 3 and path.shape[2] == 3:
            # Color image with three channels
            grayscale_image = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
        else:
            # Handle other cases
            grayscale_image = cnvrt(path)

    # convert the image to binary format
    _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

    # apply morphological operations to filter out noise
    kernel = np.ones((5, 5), np.uint8)
    seg_res = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return seg_res

#--------------------------------------------------- Edge Detection function -------------------------------------------------------------
def edge_dtct(path):
    # Image Read
    if isinstance(path, str):
        image = cv2.imread(path)
    else:
        image = path
    edge = cv2.Canny(image , 100 , 200)
    return edge

#---------------------------------------------------- Skull Removal function --------------------------------------------------------------
def skull_del(path):
    # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path
        
    # Set dimensions
    dim=(500,590)
    img = cv2.resize(img, dim)
    
    if len(img.shape) == 2:
        # Grayscale image
        gray = img
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0.7)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Color image with three channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0.7)
        
    #Threshold the image to binary using Otsu's method
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((0,0,255))
    blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
    # cv2.imshow('Blended', blended)

    ret, markers = cv2.connectedComponents(thresh)
    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
    #Get pixels which correspond to the brain
    brain_mask = markers==largest_component
    brain_out = img.copy()
    # Clearing the pixels that don"t correspond to the brain in a copy
    brain_out[brain_mask==False] = (0,0,0)
   
    return brain_out

#----------------------------------------------------- Sharpening function ----------------------------------------------------------------
def sharpen(path):
    if isinstance(path, str):
        org = cv2.imread(path)
    else:
        org = path

    if len(org.shape) == 2:
        org = cv2.cvtColor(org, cv2.COLOR_GRAY2BGR)

    sharp_kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
    sharp_image = cv2.filter2D(org, -1, sharp_kernel)
    return sharp_image

#----------------------------------------------- Histogram Equalization function ----------------------------------------------------------
def hist_equ(path):
    # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        # Grayscale image
        gray = img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Color image with three channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # creating a Histograms Equalization of an image using cv2.equalizeHist() 
    equ = cv2.equalizeHist(gray)
    
    return equ

#---------------------------------------------------- Thresholding function ---------------------------------------------------------------
def thresh(path):
    # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        # Grayscale image
        gray = img
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0.7)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Color image with three channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0.7)
        
    (T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    return thresh

def thresh_inv(path):
    # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        # Grayscale image
        gray = img
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0.7)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Color image with three channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 0.7)
        
    (T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)
    return thresh

#----------------------------------------------------- Smoothing functions ----------------------------------------------------------------
def av_blr(path):
    # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    avg = cv2.blur(img,(11,11))
    return avg

def guas_blr(path):
   # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    gaus = cv2.GaussianBlur(img,(7,7),2)
    return gaus

def median(path):
   # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    med = cv2.medianBlur(img,5)
    return med

def bilat(path):
    # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    bilateralFilter = cv2.bilateralFilter(img,9,75,75)
    return bilateralFilter

#-------------------------------------------------------- Soble function ------------------------------------------------------------------
def soble(path):
    # Image Read
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    if len(img.shape) == 2:
        # Grayscale image
        gray = img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Color image with three channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    blur = cv2.GaussianBlur(gray, (0,0), 1.3, 1.3)

    # apply sobel derivatives
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)

    # optionally normalize to range 0 to 255 for proper display
    sobelx_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
    sobely_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

    # square 
    sobelx2 = cv2.multiply(sobelx,sobelx)
    sobely2 = cv2.multiply(sobely,sobely)

    # add together and take square root
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)

    # normalize to range 0 to 255 and clip negatives
    sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
    return sobel_magnitude

#------------------------------------------------------ Laplacian function ----------------------------------------------------------------
def laplac(path):
    # Image Read
    if isinstance(path, str):
        org = cv2.imread(path)
    else:
        org = path

    if len(org.shape) == 2:
        org = cv2.cvtColor(org, cv2.COLOR_GRAY2BGR)
    laplac_filter = np.array([[0,1,0],
                              [1,-4,1],
                              [0,1,0]])
    new_img= cv2.filter2D(org, -1, laplac_filter)
    sharp_image = cv2.subtract(org, new_img)
    return sharp_image

#------------------------------------------------------- Enhance function -----------------------------------------------------------------
def enh(path):
    # Image Read
    if isinstance(path, str):
        image = Image.open(path)
    elif isinstance(path, np.ndarray):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(np.uint8(path))
   
    enhancer = ImageEnhance.Contrast(image)
    factor = 2
    image_output = enhancer.enhance(factor)
    enhanced_array = np.array(image_output)
    
    return enhanced_array

#------------------------------------------------------ Negative function -----------------------------------------------------------------
def negative(path):
    if isinstance(path, str):
        org = cv2.imread(path)
    else:
        org = path

    negative = 255 - org
    return negative