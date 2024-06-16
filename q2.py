import cv2
import numpy as np

image = cv2.imread('thermal.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# closing
kernel = np.ones((9, 3), np.uint8) #kernel must be vertically bigger
filled_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

#contours
contours, _ = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# box around each contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Result', image)

cv2.waitKey(0)
cv2.destroyAllWindows()