import cv2
import numpy as np
import os
import mediapipe as mp

folderPath = "assets"
myList = os.listdir(folderPath)
overLayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

header = overLayList[0]
brushThickness = 15
eraserThickness = 50
drawColor = (255, 255, 255)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 1280)

initHand = mp.solutions.hands
mainHand = initHand.Hands()
draw = mp.solutions.drawing_utils

def handLandmarks(colorImg):
    landmarkList = []
    landmarkPositions = mainHand.process(colorImg)
    landmarkChek = landmarkPositions.multi_hand_landmarks
    if landmarkChek:
        for hand in landmarkChek:
            for index, landmark in enumerate(hand.landmark):  # Change here
                draw.draw_landmarks(img, hand, initHand.HAND_CONNECTIONS)
                landmarkList.append([index, int(landmark.x*1280), int(landmark.y*720)])
    return landmarkList

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

xp, yp = 0, 0
while True:
    success, img = cap.read()

    # Find Hand Landmarks
    finger_landmarks = handLandmarks(img)

    # Chack which fingers are up
    if finger_landmarks:
        # Tip of index finger
        x1, y1 = finger_landmarks[8][1:]
        # Tip of middle finger
        x2, y2 = finger_landmarks[12][1:]

        # If selection mode "2 fingers are up"
        if finger_landmarks[8][2] < finger_landmarks[7][2] and finger_landmarks[12][2] < finger_landmarks[11][2]:
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
            xp, yp = 0, 0
            if y1 < 125:
                if 900 < x1 < 1020:
                    # blue
                    header = overLayList[4]
                    drawColor = (230, 158, 63)
                if 700 < x1 < 800:
                    # black
                    header = overLayList[3]
                    drawColor = (51, 51, 51)
                if 500 < x1 < 600:
                    # green
                    header = overLayList[2]
                    drawColor = (114, 255, 193)
                if 300 < x1 < 400:
                    # eraser
                    header = overLayList[1]
                    drawColor = (0, 0, 0)
                if x1 < 200:
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

        # If the drawing mode "index finger is up"
        if finger_landmarks[8][2] < finger_landmarks[7][2] and finger_landmarks[12][2] > finger_landmarks[11][2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            elif drawColor != (255, 255, 255):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Flip the frame and setting the header image
    img = cv2.flip(img, 1)
    img[0:125, 0:1280] = header

    cv2.imshow("Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()