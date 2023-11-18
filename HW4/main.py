import cv2 as cv
import time as t
import apply


def siftAlgo(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(img, None)
    return keyPoints, descriptors


def orbAlgo(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=1500)  # 1500 , 100 , 500 , 1000
    keyPoints, descriptors = orb.detectAndCompute(img, None)
    return keyPoints, descriptors


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
    return keyPoints, descriptors


def featureExtractionAlgo(img, algo='brief'):
    match algo:
        case 'orb':
            keyPoints, descriptors = orbAlgo(img)
        # case 'fast':
        #     keyPoints, descriptors = fastAlgo(img)
        case 'brief':
            keyPoints, descriptors = briefAlgo(img)
        case _:
            keyPoints, descriptors = siftAlgo(img)
    return keyPoints, descriptors, algo


def templateMatching(template, img):
    result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    return max_loc, max_val


def main():
    image_paths = ['assets/7.jpg', 'assets/8.jpg']  # Update with your image paths
    images = [cv.imread(img_path, cv.IMREAD_GRAYSCALE) for img_path in image_paths]

    template_path = 'assets/88.jpg'  # Update with your template image path
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

    templateKeyPoints, template_descriptors, algoName = featureExtractionAlgo(template)

    for idx, image in enumerate(images):
        imageKeyPoints, imageDescriptors, algoName = featureExtractionAlgo(image)

        bf = cv.BFMatcher()
        matches = bf.knnMatch(template_descriptors, imageDescriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        img_matches = cv.drawMatches(template, templateKeyPoints, image, imageKeyPoints, good_matches, None,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow(f'Matches for Image {idx + 1}', img_matches)
        cv.imwrite(f'results/{algoName}_{t.time()}.png', img_matches)

    cv.waitKey(2000)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
    apply.call()
