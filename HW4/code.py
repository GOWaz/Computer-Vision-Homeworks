import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('assets/4.jpg')
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray = cv.resize(gray, (400, 400))
cv.imshow('img', gray)
# cv.waitKey(0)

# 1- Choose an object: we chose the eyes
height, width = gray.shape
middle_row = height // 2
middle_col = width // 2

region_size = 70

start_row = middle_row - region_size // 2
end_row = middle_row + region_size // 2
start_col = middle_col - region_size // 2
end_col = middle_col + region_size // 2

template = gray[start_row:end_row, start_col:end_col]
plt.imshow(template, cmap='gray')
plt.title('Selected Template')
plt.show()

# 2- Extract important features: wh used Sift Features
sift = cv.SIFT_create()
# kp = sift.detect(gray, None)
# img=cv.drawKeypoints(gray ,kp ,gray ,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imshow('image-with-keypoints', img)
# cv.waitKey(0)
keyPoints, descriptors = sift.detectAndCompute(gray, None)
image_with_keyPoints = cv.drawKeypoints(gray, keyPoints, None)
cv.imshow('Image with KeyPoints', image_with_keyPoints)
cv.waitKey(0)
