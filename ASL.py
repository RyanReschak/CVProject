#ASL PORTION
import cv2
import numpy as np
from sign_language_identifier.hand_track import handTrack
from sign_language_identifier.sign_language_nn import SLNN

characters = ["A", "B", "C", "D", "E", "F", "G",
                  "H", "I","J", "K", "L", "M", "N", "O",
                  "P", "Q", "R", "S", "T", "U", "V",
                  "W", "X", "Y", "Z"]

def loadModel():
    hands = handTrack()
    model = SLNN()
    model.load_weights()
    return hands, model

def wordTrack(hands, model):
    cam = cv2.VideoCapture(0)


    LEFT_HAND = 0
    RIGHT_HAND = 1
    #letter Q isn't commonly used so have three of them signal wanting to exit
    numQs = 0

    preference_hand = RIGHT_HAND
    previous_char = '0'
    numChar = 0
    numFalse = 0
    full_word = ""
    while True:
        success, img = cam.read()

        hands.findHands(img, draw=False)
        num_hands = hands.num_visible_hands
        handLM = []

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

            hand_img = hand_img[hand_img_dims[1]:hand_img_dims[3], hand_img_dims[0]:hand_img_dims[2]]
            hand_img = cv2.resize(hand_img, (28, 28))

            hand_img = np.array(hand_img) / 255.0
            hand_img = np.reshape(hand_img, (-1, 28, 28, 1))

            predict_arr = model.predict(hand_img)[0]
            predicted_char = characters[np.argmax(predict_arr)]

            if predicted_char == previous_char:
                numChar += 1
            else:
                numFalse += 1
            previous_char = predicted_char
            img = cv2.putText(img, predicted_char,
                                  (hand_img_dims[2], hand_img_dims[3]), color=(0, 255, 0),
                                  fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)
            if numChar > 15:
                full_word += predicted_char # add new predicted letter to word
                if predicted_char == 'Q':
                    numQs += 1
                    if numQs == 3: #return word if number of Qs is 3
                        return full_word[:len(full_word)-2]
                numChar = 0
                numFalse = 0

        img = cv2.putText(img, full_word,
                          (100, 100), color=(0, 255, 0),
                          fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)
        cv2.imshow("Camera", img)

        if (cv2.waitKey(25) & 0xFF == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()
    return full_word