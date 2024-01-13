import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import airsim
import time  # Import the time module

# Initialize AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Initialize hand detection and classification modules
cap = cv2.VideoCapture(0)  # Tello video stream
offset = 20
imgSize = 300
labels = ["Down", "Up", "Right", "Back", "Front", "Left"]

def predict_gesture(frame, detector, classifier):
    # Hand detection
    hands = detector.findHands(frame, draw=False)
    
    prediction = ""  # Assign a default value
    index = -1  # Assign a default value
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

    return prediction, index

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("D:/Python projects/Drone Project/hand_gesture_model01.h5", "D:/Python projects/Drone Project/Labels.txt")

next_detection_time = time.time() + 5  # Initialize the next detection time with a 5-second delay


counter = [0,0,0,0,0,0]
while True:
    ret, frame = cap.read()
    
    # Check if it's time for the next detection
    if time.time() >= next_detection_time:
        # Recognize hand gestures
        gesture, index = predict_gesture(frame, detector, classifier)
        
        # Reset the next detection time
        next_detection_time = time.time() + 2
        
        label = labels[index]
        
        # Display the text on the screen
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        
        label = ""
        if counter[index] < 5:
            if index == 0:
                counter[0] += 1
                client.moveByVelocityAsync(0, 0, 1, duration=1.0)
            elif index == 1:
                counter[1] += 1
                client.moveByVelocityAsync(0, 0, -1, duration=1.0)
            elif index == 2:
                counter[2] += 1
                client.moveByVelocityAsync(0, 1, 0, duration=1.0)
            elif index == 3:
                counter[3] += 1
                client.moveByVelocityAsync(1, 0 , 0, duration=1.0)
            elif index == 4:
                counter[4] += 1
                client.moveByVelocityAsync(-1, 0 , 0, duration=1.0)
            elif index == 5:
                counter[5] += 1
                client.moveByVelocityAsync(0, -1, 0, duration=1.0)    
        else:
            client.landAsync()
    # Display the video feed
    cv2.imshow('Tello Video', frame)

    # Exit the program on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
# Release resources and land the drone
cap.release()
cv2.destroyAllWindows()
print(counter[0],counter[1],counter[2],counter[3],counter[4],counter[5])

# Release control
client.armDisarm(False)
client.enableApiControl(False)
