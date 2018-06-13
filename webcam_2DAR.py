import cv2
import imutils
from ar_overlay_2d import AR2D

# read in images to be used for query and overlay
query_image = cv2.imread('images/office_pic.png')
ar_image = cv2.imread('images/smash_box_art.png')

# init ar 2d overlay
ar2d = AR2D(query_image, ar_image, min_match_count=40)

# open webcam
camera = cv2.VideoCapture(0)

# loop over the frames of the video
while True:
    # grab the current frame
    grabbed, frame = camera.read()

    frame = imutils.resize(frame, width=1000)

    # break if frame not grabbed
    if not grabbed:
        break

    # apply 2d ar overlay
    ar_frame = ar2d.ar_2d_overlay(frame)

    # display results
    cv2.imshow('AR 2D (press Q to quit)', ar_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
