# Emma Ingram and Ryan Reschak
# ArUco Marker Detection

import cv2
import sys
import numpy as np


def aruco(img, draw=False):
    # detect markers
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    corners, ids, _ = cv2.aruco.detectMarkers(image=img, dictionary=arucoDict)
    if ids is not None:
        f = 675.0
        K = np.array([[f, 0, img.shape[1]/2],
                      [0, f, img.shape[0]/2],
                      [0, 0, 1.0]])
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners=corners, markerLength=2.0,
            cameraMatrix=K, distCoeffs=np.zeros(8))
        rvec_m_c = rvecs[0]
        tm_c = tvecs[0]

        if (draw):
            cv2.aruco.drawDetectedMarkers(image=img, corners=corners, ids=ids,
                                      borderColor=(0, 0, 255))
            #cv2.aruco.drawAxis(image=img, cameraMatrix=K, distCoeffs=np.zeros(4),
            #               rvec=rvecs, tvec=tvecs, length=1.0)

        return corners, ids, rvec_m_c, tm_c

    return None, None, None, None