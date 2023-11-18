import cv2 as cv
import numpy as np

# Harris Corner
img = cv.imread("assets/Cybertruck.png")
cv.imshow("show", img)
cv.waitKey(2000)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv.cornerHarris(gray, 8, 3, 0.04)

dst = cv.dilate(dst, None)

img[dst > 0.001 * dst.max()] = [255, 0, 255]  # 0.002 , 0.001 , 0.0005 , 0.0001

cv.imshow("result", img)
cv.waitKey(2000)

# Shi-Tomasi Corner
corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 1)
# (100 , 0.01) , (500 , 0.01) , (1000 , 0.01)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, (255, 0, 255), -1)

cv.imshow("result", img)
cv.waitKey(2000)

# Scale Invariant Feature Transform (SIFT)
sift = cv.SIFT_create()
keyPoints = sift.detect(gray, None)
cv.imshow("result", cv.drawKeypoints(img, keyPoints, None, (255, 0, 255)))
cv.waitKey(2000)

# # Speeded Up Robust Features (SURF) (not working)
# surf = cv.SURF_create()
#
# keyPoints = surf.detect(gray, None)
# cv.imshow("result", cv.drawKeyPoints(img, keyPoints, None, (255, 0, 255)))

# Oriented FAST and Rotated BRIEF (ORB)
orb = cv.ORB_create(nfeatures=1500)  # 1500 , 100 , 500 , 1000
keyPoints, descriptors = orb.detectAndCompute(img, None)
cv.imshow('result', cv.drawKeypoints(img, keyPoints, None, (255, 0, 255)))
cv.waitKey(2000)

# FAST (Features from Accelerated Segment Test):
fast = cv.FastFeatureDetector_create()
keyPoints = fast.detect(gray, None)
# cv.imshow('result', cv.drawKeypoints(img, keyPoints, None, (255, 0, 255)))
# cv.waitKey(2000)

# BRIEF (Binary Robust Independent Elementary Features)
# Compute BRIEF descriptors for the keyPoints
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
keyPoints, _ = brief.compute(gray, keyPoints)
cv.imshow('result', cv.drawKeypoints(img, keyPoints, None, (255, 0, 255)))
cv.waitKey(2000)
