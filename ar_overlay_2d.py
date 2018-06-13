import cv2
import numpy as np
import imutils.feature.factories as kp_factory


class AR2D:
    def __init__(self, query_image, ar_image, min_match_count=10):
        self.query_image = query_image
        self.ar_image = ar_image
        self.min_match_count = min_match_count

        self.query_corners = self.__get_image_corners(self.query_image)
        self.ar_corners = self.__get_image_corners(self.ar_image)

        self.query_gray = cv2.cvtColor(self.query_image, cv2.COLOR_BGR2GRAY)
        self.sift = kp_factory.FeatureDetector_create('SIFT')
        self.query_kps, self.query_kp_desc = self.sift.detectAndCompute(self.query_gray, None)

        self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

    @staticmethod
    def __get_image_corners(image):
        h, w = image.shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        return corners

    def ar_2d_overlay(self, target_image):
        """2D AR Overlay of a Query in a Target Image

        Find a query in a target image and overlay the query image's region with another image for a simple
        'augmented reality' example

        :param target_image: image that ar overlay will performed in (if query image is matched)
        :return: target_image with ar_image overlaying found query_image; if query_image not found then an unchanged
                 target_image is returned
        """
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        # find the target image's keypoints and descriptors with SIFT
        target_kps, target_kp_desc = self.sift.detectAndCompute(target_gray, None)
        # match keypoints of query <-> target
        matches = self.flann.knnMatch(self.query_kp_desc, target_kp_desc, k=2)

        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > self.min_match_count:
            # subset all matched points
            src_pts = np.float32([self.query_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([target_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # black out query image found in target image
            h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            target_corners = cv2.perspectiveTransform(self.query_corners, h_mat)
            cv2.fillConvexPoly(target_image, np.int32(target_corners), 0, 16)

            # warp ar image to fit in blacked out portion
            h_mat, mask = cv2.findHomography(self.ar_corners, target_corners, cv2.RANSAC, 5.0)
            ar_image_warped = cv2.warpPerspective(self.ar_image,
                                                  h_mat,
                                                  target_image.shape[:2][::-1])

            # overlay warped ar image into target image
            result = target_image + ar_image_warped

        else:
            # return unchanged target image if query image not found
            print("Not enough matches are found - %d/%d" % (len(good_matches), self.min_match_count))
            result = target_image

        return result


if __name__ == '__main__':
    query_bgr = cv2.imread('images/book_query_image.png')
    target_bgr = cv2.imread('images/book_target_image.png')
    ar_bgr = cv2.imread('images/smash_box_art.png')

    ar2d = AR2D(query_bgr, ar_bgr)

    overlaid = ar2d.ar_2d_overlay(target_bgr)
    cv2.imwrite('readme/example.png', overlaid)
    cv2.imshow('ar overlay', overlaid)
    cv2.waitKey(0)
