import cv2
import math
import numpy as np

def splitRGB(img):
	B_map, G_map, R_map= cv2.split(img)
	zero_channel = np.zeros(img.shape[:2], dtype = "uint8")

	return cv2.merge([zero_channel, zero_channel, R_map]), cv2.merge([zero_channel, G_map, zero_channel]), cv2.merge([B_map, zero_channel, zero_channel])

def splitHSV(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	H_map, S_map, V_map = cv2.split(hsv_img)

	return H_map, S_map, V_map


def bilinear_resize(image, height, width):

	img_height, img_width = image.shape[:2]

	resized = np.zeros((height, width, 3), np.uint8)
	
	x_ratio = float(img_width - 1) / float(width - 1)
	y_ratio = float(img_height - 1) / float(height - 1)
	print( x_ratio)
	print( y_ratio )
	for i in range(height):
		for j in range(width):

			x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
			x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)
	
			x_weight = (x_ratio * j) - x_l
			y_weight = (y_ratio * i) - y_l
	
			a = image[y_l, x_l]
			b = image[y_l, x_h]
			c = image[y_h, x_l]
			d = image[y_h, x_h]
	
			pixel = a * (1 - x_weight) * (1 - y_weight) \
				+ b * x_weight * (1 - y_weight) + \
				c * y_weight * (1 - x_weight) + \
				d * x_weight * y_weight
	
			resized[i][j] = pixel
	
	return resized

def resize(img, size):
	height = int(np.around(img.shape[0] * size))
	weight = int(np.around(img.shape[1] * size))
	print("The image size weigh {weight} and heigh {height}".format(weight=weight, height=height))

	return bilinear_resize(img, height, weight)

class MotionDetect(object):
	"""docstring for MotionDetect"""
	def __init__(self, shape):
		super(MotionDetect, self).__init__()

		self.shape = shape
		self.avg_map = np.zeros((self.shape[0], self.shape[1], self.shape[2]),  np.uint8)
		self.alpha = 0.8 # you can ajust your value
		self.threshold = 40 # you can ajust your value

		print("MotionDetect init with shape {}".format(self.shape))

	def getMotion(self, img):
		assert img.shape == self.shape, "Input image shape must be {}, but get {}".format(self.shape, img.shape)

		# Extract motion part (hint: motion part mask = difference between image and avg > threshold)
		moving = cv2.absdiff(self.avg_map.astype(np.uint8), img.astype(np.uint8))

		# Mask out unmotion part (hint: set the unmotion part to 0 with mask)
		moving_map = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)

		height = img.shape[0]
		width = img.shape[1]

		for i in range(0, height):
			for j in range(0, width):
				if moving[i, j].sum() < self.threshold:
					moving_map[i, j] = [0, 0, 0]
				else:
					moving_map[i, j] = img[i, j]
		
		# Update avg_map
		self.avg_map = self.avg_map * self.alpha + img * (1 -self.alpha)

		return moving_map

class BonusMotionDetect(MotionDetect):
	def __init__(self, shape):
		self.shape = shape
		self.avg_map = np.zeros((self.shape[0], self.shape[1]),  np.uint8)
		self.alpha = 0.8 # you can ajust your value
		self.threshold = 40 # you can ajust your value

		print("MotionDetect init with shape {}".format(self.shape))

	def getMotion(self, img):
		
		# To gray 
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Gaussian Blur
		gray = cv2.GaussianBlur(gray, (3, 3), 0)

		gray = cv2.erode(gray, (3, 3), iterations = 1)
		# Extract motion part (hint: motion part mask = difference between image and avg > threshold)
		moving = cv2.absdiff(self.avg_map.astype(np.uint8), gray.astype(np.uint8))

		# Mask out unmotion part (hint: set the unmotion part to 0 with mask)
		moving_map = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)

		height = gray.shape[0]
		width = gray.shape[1]

		for i in range(0, height):
			for j in range(0, width):
				if moving[i, j].sum() < self.threshold:
					moving_map[i, j] = 0
				else:
					moving_map[i, j] = gray[i, j]
		
		moving_map=cv2.dilate(moving_map, (3, 3), iterations = 1)

		# Update avg_map
		self.avg_map = self.avg_map * self.alpha + gray * (1 -self.alpha)

		return moving_map
def rgb_hsv():
	# ------------------ #
	#     RGB & HSV      #
	# ------------------ #
	name = "../data.png"
	img = cv2.imread(name)
	if img is not None:
		print("Reading {} success. Image shape {}".format(name, img.shape))
	else:
		print("Faild to read {}.".format(name))
	
	R_map, G_map, B_map = splitRGB(img)
	H_map, S_map, V_map = splitHSV(img)
	
	
	cv2.imwrite('data_R.png', R_map)
	cv2.imwrite('data_G.png', G_map)
	cv2.imwrite('data_B.png', B_map)
	cv2.imwrite('data_H.png', H_map)
	cv2.imwrite('data_S.png', S_map)
	cv2.imwrite('data_V.png', V_map)

def interpolation():
	# ------------------ #
	#   Interpolation    #
	# ------------------ #
	name = "../data.png"
	img = cv2.imread(name)
	if img is not None:
		print("Reading {} success. Image shape {}".format(name, img.shape))
	else:
		print("Faild to read {}.".format(name))

	height, width, channel = img.shape
	img_big = resize(img, 2)
	img_small = resize(img, 0.5)
	img_big_cv = cv2.resize(img, (width*2, height*2))
	img_small_cv = cv2.resize(img, (width//2, height//2))

	cv2.imwrite('data_2x.png', img_big)
	cv2.imwrite('data_0.5x.png', img_small)
	cv2.imwrite('data_2x_cv.png', img_big_cv)
	cv2.imwrite('data_0.5x_cv.png', img_small_cv)

def motion():
	# ------------------ #
	#  Video Read/Write  #
	# ------------------ #
	name = "../data.mp4"
	# Input reader
	cap = cv2.VideoCapture(name)
	fps = cap.get(cv2.CAP_PROP_FPS)
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

	# Output writer
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output1.avi', fourcc, fps, (w, h), True)
	bout = cv2.VideoWriter('output2.avi', fourcc, fps, (w, h), False) 

	# Motion detector
	mt = MotionDetect(shape=(h,w,3))
	bmt = BonusMotionDetect(shape=(h,w))

	# Read video frame by frame
	while True:
		# Get 1 frame
		success, frame = cap.read()

		if success:
			motion_map = mt.getMotion(frame)
			bonus_motion_map = bmt.getMotion(frame)

			# Write 1 frame to output video
			out.write(motion_map)
			bout.write(bonus_motion_map)
		else:
			break

	# Release resource
	cap.release()
	out.release()
	bout.release()

if __name__ == '__main__':
	rgb_hsv()
	interpolation()
	motion()
