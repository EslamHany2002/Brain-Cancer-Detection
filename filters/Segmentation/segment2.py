import cv2
import numpy as np

def read_image(file_path):
    # read the image in grayscale format
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # check if the image has been successfully read
    if image is None:
        print("Error: Unable to read the image.")
        return None

    return image

def segment_image(image):
    # convert the image to binary format
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # apply morphological operations to filter out noise
    kernel = np.ones((5, 5), np.uint8)
    filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return filtered_image

# file path to the image
image_file_path = "m (7).jpg"

# read the image
image = read_image(image_file_path)

# check if the image has been successfully read
if image is not None:
    # segment the image
    segmented_image = segment_image(image)

    # display the segmented image
    cv2.imshow("Segmented Image", segmented_image)

    # wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
