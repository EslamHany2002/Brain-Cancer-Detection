import cv2

def extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return keypoints, descriptors, image_with_keypoints

# Example usage:
image_path = 'imagw3.jpg'
keypoints, descriptors, image_with_keypoints = extract_features(image_path)

# Display the image with keypoints
cv2.imshow('Image with Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2

# # Load MRI image
# image = cv2.imread('imagw3.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Initialize SIFT detector
# sift = cv2.SIFT_create()

# # Detect keypoints and compute descriptors
# keypoints, descriptors = sift.detectAndCompute(gray, None)

# # Draw keypoints on the original image
# image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# # Show the image with keypoints
# cv2.imshow('MRI Image with Keypoints', image_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
