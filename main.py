import cv2
import numpy as np
import sys

#import files
sys.path.append('/sign_language_identifier/main.py')
import aruco
import main as cam



def cameraRun():
    vid = cv2.VideoCapture(0)
    while (True):

        got_img, frame = vid.read()
        if not got_img:
            break

        corners, ids, rvec, tvec = aruco.aruco(frame)
        if ids is not None:
            # draw pose on markers
            f = 675.0
            K = np.array([[f, 0, frame.shape[1]/2],
                          [0, f, frame.shape[0]/2],
                          [0, 0, 1.0]])
            cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids,
                                          borderColor=(0, 0, 255))
            cv2.aruco.drawAxis(image=frame, cameraMatrix=K, distCoeffs=np.zeros(4),
                               rvec=rvec, tvec=tvec, length=1.0)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        #hit q to exit
        #we're 'anding' in this to grab the last 8 digits to compare to ord('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cam.main()

