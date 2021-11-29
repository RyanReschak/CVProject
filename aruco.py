# Emma Ingram
# ArUco Marker Detection

import cv2
import sys
import numpy as np


def aruco(img):
    # detect markers
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    corners, ids, _ = cv2.aruco.detectMarkers(image=img, dictionary=arucoDict)
    if ids is not None:
        f = 675.0
        K = np.array([[f, 0, frame.shape[1]/2],
                      [0, f, frame.shape[0]/2],
                      [0, 0, 1.0]])
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners=corners, markerLength=2.0,
            cameraMatrix=K, distCoeffs=np.zeros(8))
        rvec_m_c = rvecs[0]
        tm_c = tvecs[0]

        return corners, ids, rvec_m_c, tm_c

    return None, None, None, None