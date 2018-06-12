import cv2
import numpy as np
import imutils.feature.factories as kp_factory


def ar_overlay(query_image, target_image, ar_image, min_match_count=10):
    """2D AR Overlay of a Query in a Target Image

    Find a query in a target image and overlay the query image's region with another image for a simple
    'augmented reality' example

    :param query_image: image to be found in target image
    :param target_image: image that ar overlay will performed in (if query image is matched)
    :param ar_image: if query image is found then this image will be overlay the found query in the target image
    :param min_match_count: minimum number of keypoints needed to confirm query is found
    :return: target_image with ar_image overlaying found query_image; if query_image not found then an unchanged
             target_image is returned
    """
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = kp_factory.FeatureDetector_create('SIFT')

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    index_params = {'algorithm': 0, 'trees': 5}
    search_params = {'checks': 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        # subset all matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # black out query image found in target image
        h, w = query_gray.shape[:2]
        query_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        target_corners = cv2.perspectiveTransform(query_corners, h_mat)
        cv2.fillConvexPoly(target_image, np.int32(target_corners), 0, 16)

        # warp ar image to fit in blacked out portion
        h, w = ar_image.shape[:2]
        ar_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        h_mat, mask = cv2.findHomography(ar_corners, target_corners, cv2.RANSAC, 5.0)
        ar_image_warped = cv2.warpPerspective(ar_image,
                                              h_mat,
                                              target_image.shape[:2][::-1])

        # overlay warped ar image into target image
        result = target_image + ar_image_warped

    else:
        # return unchanged target image if query image not found
        print("Not enough matches are found - %d/%d" % (len(good), min_match_count))
        result = target_image

    return result


if __name__ == '__main__':
    query_bgr = cv2.imread('images/query_image.png')
    target_bgr = cv2.imread('images/target_image.png')
    ar_bgr = cv2.imread('images/smash_box_art.png')

    overlaid = ar_overlay(query_bgr, target_bgr, ar_bgr)
    cv2.imwrite('readme/example.png', overlaid)
    cv2.imshow('ar overlay', overlaid)
    cv2.waitKey(0)
