import cv2 as cv
import numpy as np

# images with high details (picture of Italy and the other one for Rome)
img1 = cv.imread('asset/1.jpg')
img1 = cv.resize(img1, (600, 600))
img2 = cv.imread('asset/2.jpg')
img2 = cv.resize(img2, (600, 600))
cv.imshow('Italy', img2)
cv.imshow('Rome', img1)
cv.waitKey(0)
cv.imwrite('saved/img1.png', img1)
cv.imwrite('saved/img2.png', img2)

# Blur Filters:
# 1- Gaussian Blur:
gaussian_blur1 = cv.GaussianBlur(img1, (3, 3), cv.BORDER_DEFAULT)
gaussian_blur2 = cv.GaussianBlur(img2, (3, 3), cv.BORDER_DEFAULT)

cv.imshow('Gaussian Blur1', gaussian_blur1)
cv.imshow('Gaussian Blur2', gaussian_blur2)
cv.waitKey(2000)
cv.imwrite('saved/gaussian_blur1.png', gaussian_blur1)
cv.imwrite('saved/gaussian_blur2.png', gaussian_blur2)


# 2- median Blur:
median_blur1 = cv.medianBlur(img1, 3)
median_blur2 = cv.medianBlur(img2, 3)

cv.imshow('Median Blur1', median_blur1)
cv.imshow('Median Blur2', median_blur2)
cv.waitKey(2000)
cv.imwrite('saved/median_blur1.png', median_blur1)
cv.imwrite('saved/median_blur2.png', median_blur2)

# 3-Box Blur (Average Blur):
box_blur1 = cv.blur(img1, (3, 3))
box_blur2 = cv.blur(img2, (3, 3))

cv.imshow('Box Blur1', box_blur1)
cv.imshow('Box Blur2', box_blur2)
cv.waitKey(2000)
cv.imwrite('saved/box_blur1.png', box_blur1)
cv.imwrite('saved/box_blur2.png', box_blur2)


# 4-Motion Blur:
Kernel_size = 15
Kernel = np.zeros((Kernel_size, Kernel_size))
Kernel[int((Kernel_size - 1) / 2), :] = np.ones(Kernel_size)
Kernel /= Kernel_size

motion_blur1 = cv.filter2D(img1, -1, Kernel)
motion_blur2 = cv.filter2D(img2, -1, Kernel)

cv.imshow('Motion Blur1', motion_blur1)
cv.imshow('Motion Blur2', motion_blur2)
cv.waitKey(2000)
cv.imwrite('saved/motion_blur1.png', motion_blur1)
cv.imwrite('saved/motion_blur2.png', motion_blur2)

# Canny Filter:
# For Gaussian:
canny_filter_for_gaussian_blur1 = cv.cvtColor(gaussian_blur1, cv.COLOR_BGR2GRAY)
canny_filter_for_gaussian_blur2 = cv.cvtColor(gaussian_blur2, cv.COLOR_BGR2GRAY)

canny_filter_for_gaussian_blur1 = cv.Canny(canny_filter_for_gaussian_blur1, 125, 175)
canny_filter_for_gaussian_blur2 = cv.Canny(canny_filter_for_gaussian_blur2, 125, 175)

cv.imshow('canny_filter_for_gaussian_blur1', canny_filter_for_gaussian_blur1)
cv.imshow('canny_filter_for_gaussian_blur2', canny_filter_for_gaussian_blur2)
cv.waitKey(2000)
cv.imwrite('saved/canny_filter_for_gaussian_blur1.png', canny_filter_for_gaussian_blur1)
cv.imwrite('saved/canny_filter_for_gaussian_blur2.png', canny_filter_for_gaussian_blur2)

# For Median:
canny_filter_for_median_blur1 = cv.cvtColor(median_blur1, cv.COLOR_BGR2GRAY)
canny_filter_for_median_blur2 = cv.cvtColor(median_blur2, cv.COLOR_BGR2GRAY)

canny_filter_for_median_blur1 = cv.Canny(canny_filter_for_median_blur1, 125, 175)
canny_filter_for_median_blur2 = cv.Canny(canny_filter_for_median_blur2, 125, 175)

cv.imshow('canny_filter_for_median_blur1', canny_filter_for_median_blur1)
cv.imshow('canny_filter_for_median_blur2', canny_filter_for_median_blur2)
cv.waitKey(2000)
cv.imwrite('saved/canny_filter_for_median_blur1.png', canny_filter_for_median_blur1)
cv.imwrite('saved/canny_filter_for_median_blur2.png', canny_filter_for_median_blur2)

# For Box Blur:
canny_filter_for_box_blur1 = cv.cvtColor(box_blur1, cv.COLOR_BGR2GRAY)
canny_filter_for_box_blur2 = cv.cvtColor(box_blur2, cv.COLOR_BGR2GRAY)

canny_filter_for_box_blur1 = cv.Canny(canny_filter_for_box_blur1, 125, 175)
canny_filter_for_box_blur2 = cv.Canny(canny_filter_for_box_blur2, 125, 175)

cv.imshow('canny_filter_for_box_blur1', canny_filter_for_box_blur1)
cv.imshow('canny_filter_for_box_blur2', canny_filter_for_box_blur2)
cv.waitKey(2000)
cv.imwrite('saved/canny_filter_for_box_blur1.png', canny_filter_for_box_blur1)
cv.imwrite('saved/canny_filter_for_box_blur2.png', canny_filter_for_box_blur2)

# For Motion Blur:
canny_filter_for_motion_blur1 = cv.cvtColor(motion_blur1, cv.COLOR_BGR2GRAY)
canny_filter_for_motion_blur2 = cv.cvtColor(motion_blur2, cv.COLOR_BGR2GRAY)

canny_filter_for_motion_blur1 = cv.Canny(canny_filter_for_motion_blur1, 125, 175)
canny_filter_for_motion_blur2 = cv.Canny(canny_filter_for_motion_blur2, 125, 175)

cv.imshow('canny_filter_for_motion_blur1', canny_filter_for_motion_blur1)
cv.imshow('canny_filter_for_motion_blur2', canny_filter_for_motion_blur2)
cv.waitKey(2000)
cv.imwrite('saved/canny_filter_for_motion_blur1.png', canny_filter_for_motion_blur1)
cv.imwrite('saved/canny_filter_for_motion_blur2.png', canny_filter_for_motion_blur2)

# Morphological Operations:
# 1-Erosion:
grey_image1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
grey_image2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

kernel1 = np.ones((5, 5), np.uint8)
erosion1 = cv.erode(grey_image1, kernel1, iterations=1)
cv.imshow("erosion1", erosion1)

kernel2 = np.ones((5, 5), np.uint8)
erosion2 = cv.erode(grey_image2, kernel2, iterations=1)
cv.imshow("erosion2", erosion2)
cv.waitKey(2000)
cv.imwrite('saved/erosion1.png', erosion1)
cv.imwrite('saved/erosion2.png', erosion2)

# 2-Dilation:
dilation1 = cv.dilate(grey_image1, kernel1, iterations=1)
cv.imshow("dilation1", dilation1)

dilation2 = cv.dilate(grey_image2, kernel2, iterations=1)
cv.imshow("dilation2", dilation2)
cv.waitKey(2000)
cv.imwrite('saved/dilation1.png', dilation1)
cv.imwrite('saved/dilation2.png', dilation2)

# 3-Opening
Opening1 = cv.morphologyEx(grey_image1, cv.MORPH_OPEN, kernel1)
cv.imshow("opening1", Opening1)

Opening2 = cv.morphologyEx(grey_image2, cv.MORPH_OPEN, kernel2)
cv.imshow("opening2", Opening2)
cv.waitKey(2000)
cv.imwrite('saved/Opening1.png', Opening1)
cv.imwrite('saved/Opening2.png', Opening2)

# 4-Closing
Closing1 = cv.morphologyEx(grey_image1, cv.MORPH_CLOSE, kernel1)
cv.imshow("closing1", Closing1)

Closing2 = cv.morphologyEx(grey_image2, cv.MORPH_CLOSE, kernel2)
cv.imshow("closing2", Closing2)
cv.waitKey(2000)
cv.imwrite('saved/Closing1.png', Closing1)
cv.imwrite('saved/Closing2.png', Closing2)

# 5-Gradient
Gradient1 = cv.morphologyEx(grey_image1, cv.MORPH_GRADIENT, kernel1)
cv.imshow("Gradient1", Gradient1)

Gradient2 = cv.morphologyEx(grey_image2, cv.MORPH_GRADIENT, kernel2)
cv.imshow("Gradient2", Gradient2)
cv.waitKey(2000)
cv.imwrite('saved/Gradient1.png', Gradient1)
cv.imwrite('saved/Gradient2.png', Gradient2)
