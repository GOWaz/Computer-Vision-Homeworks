import cv2
import numpy as np


def extract_features(image, method='sift'):
    if method.lower() == 'sift':
        detector = cv2.SIFT_create()
    elif method.lower() == 'orb':
        detector = cv2.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def template_matching(template, image):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val


def main():
    image_paths = ['C:/Users/tonyb/Desktop/Open CV/HomeWorks/Computer-Vision-Homeworks/HW4/assets/6.jpg',
                   'C:/Users/tonyb/Desktop/Open CV/HomeWorks/Computer-Vision-Homeworks/HW4/assets/7.jpg',
                   'C:/Users/tonyb/Desktop/Open CV/HomeWorks/Computer-Vision-Homeworks/HW4/assets/8.jpg']  # Update with your image paths
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_paths]

    template_path = 'C:/Users/tonyb/Desktop/Open CV/HomeWorks/Computer-Vision-Homeworks/HW4/assets/88.jpg'  # Update with your template image path
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    template_keypoints, template_descriptors = extract_features(template)

    for idx, image in enumerate(images):
        image_keypoints, image_descriptors = extract_features(image)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(template_descriptors, image_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        img_matches = cv2.drawMatches(template, template_keypoints, image, image_keypoints, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow(f'Matches for Image {idx + 1}', img_matches)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
