import cv2
from PIL import Image
import os
from reportlab.pdfgen import canvas
import time
import random
import string
import numpy as np
from HandTrackingModule import handDetector  # Assuming the provided module is saved as HandTrackingModule.py

# Initialize a flag to control screenshot saving
screenshot_taken = False


def save_canvas(imgInv, folder="SavedImages"):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Generate a unique filename using a timestamp and a random string
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    filename = f"{timestamp}-{random_str}.png"

    # Save the image in the specified folder
    cv2.imwrite(os.path.join(folder, filename), imgInv)
    print(f"Canvas saved as {filename} in {folder}")


def create_pdf_and_clear_folder(folder="SavedImages"):
    if not os.path.exists(folder):
        print("Folder does not exist.")
        return

    # List all image files in folder
    img_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    img_files = [os.path.join(folder, f) for f in img_files]  # Full paths

    if img_files:
        # Modify here to create the PDF in the current directory
        pdf_filename = "SavedImages.pdf"

        c = canvas.Canvas(pdf_filename)

        for img_file in img_files:
            with Image.open(img_file).convert("RGB") as img:
                width, height = img.width, img.height
                c.setPageSize((width, height))
                c.drawInlineImage(img_file, 0, 0, width=width, height=height)
                c.showPage()

        c.save()
        print(f"PDF created: {pdf_filename}")

        # Clear the folder by removing all images
        for f in img_files:
            os.remove(f)
        print("Folder cleared.")
    else:
        print("No images found in the folder.")


folderPath = "Overlay"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[3]


# Drawing settings
brushThickness = 15
eraserThickness = 50
drawColor = (255, 0, 255)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Hand detector
detector = handDetector(detectionCon=0.65, maxHands=1)

# Canvas for drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

xp, yp = 0, 0

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if lmList:
        x1, y1 = lmList[8][1:]  # Tip of the index finger
        x2, y2 = lmList[12][1:]  # Tip of the middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 3.1. Selection Threshold set up
        distance, img, [x3, y3, x4, y4, cx, cy] = detector.findDistance(4, 8, img)
        closeness_threshold = 70

        if distance < closeness_threshold:  # 3.1.1. Pointer Mode - Thumb and Index close
            xp, yp = 0, 0a
            print("Pointer")
            cv2.arrowedLine(img, (x3, y3 - 25), (x4, y4 + 25), (0, 0, 0), 4, 100)
        elif fingers[1] and fingers[2]:  # 4. Selection Mode - Two fingers are up
            xp, yp = 0, 0  # Reset drawing position
            print("Selection Mode")
            if y1 < 125:
                if 0 < x1 < 250:
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                elif 325 < x1 < 433:
                    header = overlayList[0]
                    drawColor = (0, 0, 0)
                elif 435 < x1 < 563:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 565 < x1 < 688:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 690 < x1 < 818:
                    header = overlayList[3]
                    drawColor = (255, 0, 255)
            if 860 < x1 < 1030 and y1 < 125:
                if not screenshot_taken:  # Check if a screenshot has not been taken yet
                    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
                    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
                    save_canvas(imgInv)
                    screenshot_taken = True  # Set the flag to indicate a screenshot has been taken
            else:
                screenshot_taken = False
            if 1050 < x1 < 1250 and y1 < 125:
                create_pdf_and_clear_folder()
                print("pdf Made")
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        elif fingers[1] and not fingers[2]:  # 5. Drawing Mode - Index finger is up
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    # Combine images
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Show the images
    img[0:100, 0:1280] = header
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)

