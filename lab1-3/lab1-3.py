import cv2
import numpy as np

def avgFilter(img):

    img_avg = np.zeros(img.shape, np.uint8)

    result = 0
    #Like CNN pedding 1 pixel is better
    pedding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)

    # deal with filter size = 3x3
    for j in range(1, pedding.shape[0]-1):
        for i in range(1, pedding.shape[1]-1):
            for y in range(-1, 2):
                for x in range(-1, 2):
                    #Sum the 3*3 filter
                    result = result + pedding[j+y, i+x]
            img_avg[j-1][i-1] = int(result / 9)
            result = 0

    return img_avg

def midFilter(img):
    # TODO
    img_mid = np.zeros(img.shape, np.uint8)

    #Like CNN pedding 1 pixel is better
    pedding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    
    filter_array = [pedding[0][0]] * 3 * 3
  
    for j in range(1, pedding.shape[0]-1):
        for i in range(1, pedding.shape[1]-1):
            filter_array[0] = pedding[j-1, i-1]
            filter_array[1] = pedding[j, i-1]
            filter_array[2] = pedding[j+1, i-1]
            filter_array[3] = pedding[j-1, i]
            filter_array[4] = pedding[j, i]
            filter_array[5] = pedding[j+1, i]
            filter_array[6] = pedding[j-1, i+1]
            filter_array[7] = pedding[j, i+1]
            filter_array[8] = pedding[j+1, i+1]

            # sort the array
            filter_array.sort()

            # put the median number into output array
            img_mid[j-1][i-1] = filter_array[4]
   
    return img_mid

def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def edgeSharpen(img, amount = 1):
    # TODO
    if len(img.shape) > 2 :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    denoise = cv2.GaussianBlur(gray,(3,3),0)
    kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])
    pedding = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    img_edge = cv2.filter2D(pedding,-1,kernel = kernel)

    denoise = cv2.GaussianBlur(gray,(3,3),0)
    kernel_delta = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    pedding_delta = cv2.copyMakeBorder(img_edge,1,1,1,1,cv2.BORDER_REPLICATE)
    img_delta = cv2.filter2D(pedding,-1,kernel = kernel_delta)

    return img_edge, img_delta, unsharp_mask(img_edge)

# ------------------ #
#       Denoise      #
# ------------------ #
name1 = '../noise_impulse.png'
name2 = '../noise_gauss.png'
noise_imp = cv2.imread(name1, 0)
noise_gau = cv2.imread(name2, 0)

img_imp_avg = avgFilter(noise_imp)
img_imp_mid = midFilter(noise_imp)
img_gau_avg = avgFilter(noise_gau)
img_gau_mid = midFilter(noise_gau)

cv2.imwrite('img_imp_avg.png', img_imp_avg)
cv2.imwrite('img_imp_mid.png', img_imp_mid)
cv2.imwrite('img_gau_avg.png', img_gau_avg)
cv2.imwrite('img_gau_mid.png', img_gau_mid)


# ------------------ #
#       Sharpen      #
# ------------------ #
name = '../mj.tif'
img = cv2.imread(name, 0)

img_edge, img_s, img_ums = edgeSharpen(img)
cv2.imwrite('mj_edge.png', img_edge)
cv2.imwrite('mj_sharpen.png', img_s)
cv2.imwrite('mj_ums.png', img_ums)