import cv2
import pytesseract
import numpy as np

def enhance_image_and_extract_text(image):
    # Convert to HSV and create mask for blue
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Invert mask to highlight hidden text
    mask_inv = cv2.bitwise_not(mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    revealed = cv2.bitwise_and(gray, gray, mask=mask_inv)

    # Improve text visibility
    revealed = cv2.GaussianBlur(revealed, (3, 3), 0)
    _, revealed = cv2.threshold(revealed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    text = pytesseract.image_to_string(revealed)
    return text, revealed