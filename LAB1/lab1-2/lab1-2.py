import cv2
import numpy as np
import matplotlib.pyplot as plt

def gammaCorrection(img, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    # TODO
    invGamma = 1.0 / gamma
    table = np.zeros( 256, np.uint8 )
    for i in range(0, 256):
        norm_val = float(i)/255.0
        table[i] = int( 255.0 * ( norm_val ** invGamma ) )
    # Apply gamma correction using the lookup table
    # TODO
    img_g = cv2.LUT(img, table)

    return img_g

def histEq(gray):
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist / gray.size
    print(hist)
    print("---Normal---\n")
    
    # Convert the histogram to Cumulative Distribution Function
    # TODO
    cdf = hist.cumsum()
    print(cdf)
    print("---Cumulative---\n")

    # Build a lookup table mapping the pixel values [0, 255] to their new grayscale value
    # TODO
    hist_table = np.zeros(256, dtype='uint8') 
    for i in range(0, 256):
        hist_table[i] = int(cdf[i] * 255.0)
    
    # Apply histogram equalization using the lookup table
    # TODO
    img_h = np.zeros(gray.shape, np.uint8)
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            img_h[i, j] = hist_table[gray[i, j]]

    return img_h


# ------------------ #
#  Gamma Correction  #
# ------------------ #
name = "../data.mp4"
cap = cv2.VideoCapture(name)
success, frame = cap.read()
if success:
    print("Success reading 1 frame from {}".format(name))
else:
    print("Faild to read 1 frame from {}".format(name))
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img_g1 = gammaCorrection(gray, 0.5)
img_g2 = gammaCorrection(gray, 2)
cv2.imwrite('gray.png', gray)
cv2.imwrite('data_g0.5.png', img_g1)
cv2.imwrite('data_g2.png', img_g2)

# ------------------------ #
#  Histogram Equalization  #
# ------------------------ #
name = "../hist.png"
img = cv2.imread(name, 0)

img_h = histEq(img)
img_h_cv = cv2.equalizeHist(img)
cv2.imwrite("hist_h.png", img_h)
cv2.imwrite("hist_h_cv.png", img_h_cv)

# save histogram
plt.figure(figsize=(18, 6))
plt.subplot(1,3,1)
plt.bar(range(1,257), cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(-1))
plt.subplot(1,3,2)
plt.bar(range(1,257), cv2.calcHist([img_h], [0], None, [256], [0, 256]).reshape(-1))
plt.subplot(1,3,3)
plt.bar(range(1,257), cv2.calcHist([img_h_cv], [0], None, [256], [0, 256]).reshape(-1))
plt.savefig('hist_plot.png')