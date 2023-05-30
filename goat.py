import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from googletrans import Translator
import re
import tkinter as Tk



# imgsrc = "C:/Users/Shyaam/Downloads/sample.jpg"
imgsrc = "E:/VIT/Tarp/jero.jpg"
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


pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
text = pytesseract.image_to_string(image) #preprocessed AHE Image is called
# print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()

mystr = re.sub(r'[^\w\s]', '', text)
split_sentence = mystr.split()

print(split_sentence)

# mystr = "bloody sweet"

translator = Translator()
translated_word = translator.translate(mystr, src='en', dest = 'ta')
print(translated_word.text)


label = Tk.Label(None, text=translated_word.text, font = 40, fg='black')
label.pack()
label.mainloop()

# length = len(split_sentence)
# dest_lan = 'ta'
# for j in range(length):
#     x = split_sentence[j]
#     translated_word = translator.translate(x, src='en', dest = dest_lan)
#     print(translated_word.text)
#     print(" ")