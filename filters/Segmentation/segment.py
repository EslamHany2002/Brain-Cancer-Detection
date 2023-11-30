import cv2

def segment_image(image):
  # Convert the image to grayscale.
  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply a threshold to the image.
  thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

  # Find the contours in the thresholded image.
  contours = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

  # Draw the contours on the original image.
  segmented_image = image.copy()
  cv2.drawContours(segmented_image, contours, -1, (0, 0, 255), 2)

  return segmented_image

# Example usage:

input_image = cv2.imread("Tr-me_0080.jpg")

# Segment the image.
segmented_image = segment_image(input_image)

# Save the converted image.
cv2.imshow("segmented", segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()