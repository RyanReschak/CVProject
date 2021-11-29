import cv2
import aruco
import numpy as np
from sign_language_identifier.sign_language_nn import SLCNN
from sign_language_identifier.hand_track import handTrack

def cameraRun():
    model = SLCNN()
    model.load_weights("sign_language_identifier/weights_slnn4.w")
    tracker = handTrack()
    vid = cv2.VideoCapture(0)

    while (True):

        got_img, frame = vid.read()
        if not got_img:
            break

        corners, ids, rvec, tvec = aruco.aruco(frame, draw=True)

        tracker.findHands(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        #hit q to exit
        #we're 'anding' in this to grab the last 8 digits to compare to ord('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cameraRun()

