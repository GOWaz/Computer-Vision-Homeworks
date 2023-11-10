import cv2 as cv
import numpy as np
import time as tm

t = tm.time()


# Standard Hough transform:
def standard_hough_transform(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(grey, 125, 175)
    lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=200)
    # الرقم 1 هو دقة التراكم في البيكسلات

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow('Standard Hough Transform', image)
    cv.imwrite(f'output/sht${t}.jpg', image)
    cv.waitKey(2000)


# Probabilistic Hough Transform:
def probabilistic_hough_transform(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(grey, 125, 175)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('Probabilistic Hough Transform', image)
    cv.imwrite(f'output/pht${t}.jpg', image)
    cv.waitKey(2000)


def remove_text(image, maskk):
    dst = cv.inpaint(image, maskk, 3, cv.INPAINT_TELEA)
    cv.imshow("OriginalImage", image)
    # cv.imshow('mask', maskk)
    cv.imshow('dst', dst)
    cv.imwrite('output/noText.jpg', dst)
    cv.waitKey(2000)


# First Image:
image11 = cv.imread('assets/3.png')
image12 = cv.imread('assets/3.png')
# standard_hough_transform(image11)
# probabilistic_hough_transform(image12)

# Second Image:
image21 = cv.imread('assets/2.jpg')
image22 = cv.imread('assets/2.jpg')
# standard_hough_transform(image21)
# probabilistic_hough_transform(image22)

# The 4th Q
bookCover = cv.imread('assets/bookCover.jpg')
mask = cv.imread('assets/mask.jpg', 0)
# bookCover = cv.resize(bookCover, (1276, 909))
# mask = cv.resize(mask, (1276, 909))
remove_text(bookCover, mask)
