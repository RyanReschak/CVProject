from sign_language_identifier import hand_track

def draw(tracker, img):
    positions = tracker.findPosition(img)
    print(positions)
