
import numpy as np
import cv2
from time import sleep

face_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')  # Face
right_eye_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_righteye_2splits.xml')  # Right eye
left_eye_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_lefteye_2splits.xml')  # Left eye
cap = cv2.VideoCapture(0)  # Load webcam


# Check if the webcam is opened correctly

if not cap.isOpened():

    raise IOError("Cannot open webcam")

while True:

    # Frame stuff
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Variables
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Gray scale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5)  # Reads Gray Scale

    # data from screen displayed onto console
    for (x, y, w, h) in faces:

        # Print data
        print(x, y, w, h)

        # Variables

        # Take image
        roi_grey = gray[y:y + h, x:x + w]  # Takes vales for image of face only
        roi_color = frame[y:y + h, x:x + w]  # Takes vales for image of face only
        r_eye = right_eye_cascade.detectMultiScale(roi_grey)  # Color and stuff for the right eye
        l_eye = left_eye_cascade.detectMultiScale(roi_grey)  # Color and stuff for the left eye
        img_item = "my_image.png"  # Saves image
        cv2.imwrite(img_item, roi_grey)  # Displays image

        # Draw rectangle for right eye
        for (ex, ey, ew, eh) in r_eye:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Draw rectangle around eyes

        # Draw rectangle for left eye
        for (ex, ey, ew, eh) in l_eye:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 225, 0), 2)

        # Create rectangle around face
        color = (255, 0, 0)  # Blue, Green, Red. Blue
        stroke = 2  # sets thickness

        end_cord_x = x + w  # Width of box
        end_cord_y = y + h  # Height of box

        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)  # Draws Box

    # Show frame
    cv2.imshow('Frame', frame)

    # Click 'q' to quit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()