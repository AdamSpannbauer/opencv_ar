import cv2
import numpy as np
import imutils
from ar_overlay_2d import AR2D

# read in images to be used for query and overlay
query_image = cv2.imread('images/crossword_query.png')
ar_image = cv2.imread('images/smash_box_art.png')

# init ar 2d overlay
ar2d = AR2D(query_image, ar_image, min_match_count=200)

# open webcam
camera = cv2.VideoCapture(0)

# loop over the frames of the video
while True:
    # grab the current frame
    grabbed, frame = camera.read()

    frame = imutils.resize(frame, width=1000)
    frame_clone = frame.copy()

    # break if frame not grabbed
    if not grabbed:
        break

    try:
        # apply 2d ar overlay
        ar_frame = ar2d.ar_2d_overlay(frame)
    except cv2.error:
        ar_frame = frame_clone

    frame_clone = imutils.resize(frame_clone, width=500)
    ar_frame = imutils.resize(ar_frame, width=500)

    comparison = np.vstack((frame_clone, ar_frame))

    # display results
    cv2.imshow('AR 2D (press Q to quit)', comparison)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
