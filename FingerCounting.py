import cv2
import time
import os
import HandTrackingModule as htm

# wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)

# folderPath = "FingerImages"
# myList = os.listdir(folderPath) # read file path
# print(myList)
# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}') # import files
#     # print(f'{folderPath}/{imPath}')
#     overlayList.append(image) # save files
#
# print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.55)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    lmList = detector.findposition(img, draw=False)
    # print(lmList)

    if hands:  # Check if any hand was detected
        totalFingers = 0
        for handIndex, LR in enumerate(hands):
            lmList = detector.findposition(img, handNo=handIndex, draw=False)
            if lmList:  # Check if landmarks were found
                fingers = fingersUp(LR, lmList)
                totalFingers += sum(fingers)
        print(totalFingers)

        # h, w, c = overlayList[totalFingers - 1].shape
        # img[0:h, 0:w] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20, 225), (270, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)  # delay to see img 1ms


    def fingersUp(hand, lmList):
        fingers = []
        # Hands
        if hand == "Right":  # Right hands
            # Thumb
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Right Hand
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # Left Hand
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
