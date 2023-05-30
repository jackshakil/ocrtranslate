import cv2
import pytesseract
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.image as mpimg
from difflib import get_close_matches
from googletrans import Translator

img = cv2.imread("E:/VIT/Tarp/Deskewed Image.jpg")
# cv2.imshow("cropped", img)
# cv2.waitKey(0)
# plt.figure(1)
# plt.imshow(img)

# plt.title("Image extracted from Bounding Box")
# plt.show()
# cv2.imwrite("Cropped Image.jpg", img)

pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
text = pytesseract.image_to_string(img) #preprocessed AHE Image is called
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()

split_sentence = text.split(' ')
#split_sentence = map(lambda s: split_sentence.strip(), split_sentence)

m = len(split_sentence)
x = split_sentence[m-1].replace("\n", "")
split_sentence[m-1] = x
print(split_sentence)
print(x)
translator = Translator()
    
length = len(split_sentence)
dest_lan = 'ta'
#take word from user
#word = input('Enter word: ')
for j in range(length):
    x = split_sentence[j]
    translated_word = translator.translate(x, src='en', dest = dest_lan)
    print(translated_word.text)
    print(" ")