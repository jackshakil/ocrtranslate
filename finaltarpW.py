import cv2
import numpy as np
from matplotlib import pyplot as plt
# from scipy.ndimage import interpolation as inter
import pytesseract
from googletrans import Translator
import re

# raspistill -o ~/Pictures/stuff.jpg

# pre processing

imgsrc = "E:/VIT/TARP/bruh.jpg"
image = cv2.imread(imgsrc)
# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = 0.9*accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * 0.9*alpha
    
    
    #Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show() 
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

auto_result, alpha, beta = automatic_brightness_and_contrast(image)
print('alpha', alpha)
print('beta', beta)
# cv2.imshow('auto_result', auto_result)
# plt.figure()
# plt.show()
# plt.imshow(image)
# plt.figure()
# plt.show()
plt.imshow(auto_result)
plt.title('AHE Image')
plt.show()
cv2.imwrite("AHE Image.jpg", auto_result)
cv2.waitKey()

# deskew

skewsrc = "E:/VIT/TARP/AHE Image.jpg"
img = auto_result

# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle 

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# # Deskew image
# def deskew(cvImage):
#     angle = getSkewAngle(cvImage)
#     return rotateImage(cvImage, -1.0 * angle + 90)
#     #Return img

# angle = getSkewAngle(img)
# bruhh = rotateImage(img, angle)
# rot = deskew(bruhh)


# plt.imshow(rot)
# plt.title('Deskewed Image')
# plt.show()
# cv2.imwrite("Deskewed Image.jpg", rot)
# cv2.waitKey()

# # text detection


# image = rot
# # cv2.imshow("cropped", img)
# # cv2.waitKey(0)
# # plt.figure(1)
# # plt.imshow(img)

# # plt.title("Image extracted from Bounding Box")
# # plt.show()
# # cv2.imwrite("Cropped Image.jpg", img)

pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
text = pytesseract.image_to_string(auto_result) #preprocessed AHE Image is called
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()

# stray = text.split(' ')
# print(stray)

#split_sentence = map(lambda s: split_sentence.strip(), split_sentence)

mystr = re.sub(r'[^\w\s]', '', text)
split_sentence = mystr.split()

print(split_sentence)

translator = Translator()
    
length = len(split_sentence)
dest_lan = 'ta'
for j in range(length):
    x = split_sentence[j]
    translated_word = translator.translate(x, src='en', dest = dest_lan)
    print(translated_word.text)
    print(" ")