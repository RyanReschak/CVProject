import cv2

def cameraRun():
    vid = cv2.VideoCapture(0)
    while (True):

        _, frame = vid.read()

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

