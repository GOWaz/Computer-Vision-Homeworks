import numpy as np
import cv2 as cv
import time
from matplotlib import pyplot as plt


def siftAlgo(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(img, None)
    return keyPoints, descriptors, sift


def orbAlgo(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=1500)  # 1500 , 100 , 500 , 1000
    keyPoints, descriptors = orb.detectAndCompute(img, None)
    return keyPoints, descriptors, 'orb'


def fastAlgo(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    keyPoints = fast.detect(img, None)
    return keyPoints


def briefAlgo(img):
    keyPoints = fastAlgo(img)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keyPoints, descriptors = brief.compute(img, keyPoints)
    return keyPoints, descriptors, 'brief'


def call():
    MIN_MATCH_COUNT = 10
    img1 = cv.imread('assets/44.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread('assets/6.jpg', cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keyPoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # kp1, des1, algoName = siftAlgo(img1)
    # kp2, des2, algoName = siftAlgo(img2)

    # kp1, des1, algoName = orbAlgo(img1)
    # kp2, des2, algoName = orbAlgo(img2)

    # kp1, des1, algoName = briefAlgo(img1)
    # kp2, des2, algoName = briefAlgo(img2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv.imwrite(f'results/_fileApply_{time.time()}.png', img3)
    plt.imshow(img3, 'gray'), plt.show()
