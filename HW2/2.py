import cv2 as cv
import numpy as np


def convertToBin(img):
    threshold_value = 128
    _, binary_image = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY)
    return binary_image


def printImageArray(img):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                print('0', end=' ')
            else:
                print('1', end=' ')
        print()


# Read

# def printBinImageArray(img):
#     height, width = img.shape
#     threshold_value = 128
#     for i in range(height):
#         for j in range(width):
#             if img[i, j] < threshold_value:
#                 print('0', end=' ')
#             else:
#                 print('1', end=' ')
#         print()

def printBinImageArray(img, threshold):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] < threshold:
                print('0', end=' ')
            else:
                print('1', end=' ')
        print()


gray_image = cv.imread('asset/4.jpg', cv.IMREAD_GRAYSCALE)
gray_image = cv.resize(gray_image, (10, 10))
cv.imwrite('saved/gray_image.png', gray_image)

binImage = convertToBin(gray_image)
cv.imwrite('saved/binImage.png', binImage)

printImageArray(binImage)

kernel = np.ones((3, 3), np.uint8)
erosion = cv.erode(binImage, kernel, iterations=1)
print(kernel)
printBinImageArray(erosion, 255)
cv.imwrite('saved/erosion_4.png', erosion)

cv.imshow("erosion", erosion)
cv.waitKey(0)
cv.destroyAllWindows()
