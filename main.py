import aruco
import numpy as np

import ASL


if __name__ == '__main__':
    hands, model = ASL.loadModel()
    word = ASL.wordTrack(hands, model) #can be called repeatedly when needed

