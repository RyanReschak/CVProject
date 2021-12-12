#ASL PORTION
#Ryan Reschak, Emma Ingram, Julia Harvey, Lauren Loe
import cv2
import numpy as np
import aruco as ar
from sign_language_identifier.hand_track import handTrack
from sign_language_identifier.sign_language_nn import SLCNN

characters = ["A", "B", "C", "D", "E", "F", "G",
                  "H", "I","J", "K", "L", "M", "N", "O",
                  "P", "Q", "R", "S", "T", "U", "V",
                  "W", "X", "Y", "Z"]

def loadModel():
    hands = handTrack()
    model = SLCNN()
    #Version 5 is the newest and best weights
    model.load_weights("sign_language_identifier/weights_slnn5.w")
    return hands, model

def wordTrack(hands, model):
    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    out = cv2.VideoWriter('sign_language_augmented2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))

    LEFT_HAND = 0
    RIGHT_HAND = 1
    #letter Q isn't commonly used so have three of them signal wanting to exit
    numQs = 0

    preference_hand = RIGHT_HAND
    previous_char = '0'
    numChar = 0
    numFalse = 0
    full_word = ""
    predicted_char = ""

    f = 554
    K = np.array(((f, 0, int((cam.get(cv2.CAP_PROP_FRAME_WIDTH)/2))), (0, f, int((cam.get(cv2.CAP_PROP_FRAME_HEIGHT))/2)), (0, 0, 1)), dtype=np.float32)
    #Loop through the video
    while True:
        success, img = cam.read()

        hands.findHands(img, draw=False)
        num_hands = hands.num_visible_hands
        handLM = []

        # Based on the Hand find the Location of the points on the hand
        if (num_hands == 1):
            # Whichever hand is in the image
            handLM = hands.findPosition(img)
        elif (num_hands == 2):
            # Hand Preference
            handLM = hands.findPosition(img, handNum=preference_hand)

        if (handLM != []):
            if numFalse + numChar > 30: #if too many vals, remove some instances
                numFalse -= 1
                numChar -= 1

            # Dims can be negative based on the NN so if it crashes that's why
            hand_img_dims = hands.boundingBox(handLM, padding=20)
            # Projects img of hand (y start -> y end, x start -> x end)
            hand_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #Using those marker locations get the bounding box and image to feed the CNN
            hand_img = hand_img[hand_img_dims[1]:hand_img_dims[3], hand_img_dims[0]:hand_img_dims[2]]
            hand_img = cv2.resize(hand_img, (28, 28))
            hand_img = np.array(hand_img) / 255.0
            hand_img = np.reshape(hand_img, (-1, 28, 28, 1))

            #Predict Using the model
            predict_arr = model.predict(hand_img)[0]
            predicted_char = characters[np.argmax(predict_arr)]

            if predicted_char == previous_char:
                numChar += 1
            else:
                numFalse += 1
            previous_char = predicted_char
            #img = cv2.putText(img, predicted_char,
             #                     (hand_img_dims[2], hand_img_dims[3]), color=(0, 255, 0),
              #                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)
            if numChar > 20:
                full_word += predicted_char # add new predicted letter to word
                if predicted_char == 'Q':
                    #numQs += 1
                    if numQs == 3: #return word if number of Qs is 3
                        return full_word[:len(full_word)-2]
                numChar = 0
                numFalse = 0
            corners, ids, rvec_m_c, tm_c = ar.aruco(img, draw=True)

            #Draw the Prediction on the Marker in the Image
            if ids is not None:

                pImg, J = cv2.projectPoints(objectPoints=np.array((1,0.5,0),dtype=np.float32), rvec=rvec_m_c, tvec=tm_c, cameraMatrix=K,
                                        distCoeffs=None)
                #print(pImg[0][0])
                letter = ""
                if full_word != "":
                    letter = full_word[-1]
                cv2.putText(img, predicted_char,
                              tuple(np.int32(pImg[0][0])), color=(0, 0, 255),
                              fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5,thickness=3)
        #Finally Output the Resulting Image
        cv2.imshow("Camera", img)
        #Save Output File
        out.write(img)
        #Q is to Quit
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()
    return full_word
