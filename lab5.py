import cv2
import json

color_code = [(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),(255,0,255),(0,0,0),(255,255,255),(128,128,128)]
link = [-1,0,1,2,3,1,5,6,1,8,9,10,8,12,13,-1,-1,-1,-1,14,14,14,11,11,11]

def process_img(id,frame):
	str_file = 'json/video_{:0>12d}_keypoints.json'.format(id)
	with open(str_file) as json_file:
		data = json.load(json_file)
		people_list = data['people']
		color_id = 0;
		for people in people_list:
			point_list = people['pose_keypoints_2d']
			for index in range(0,len(point_list),3):
				if point_list[index+2] != 0:
					frame = cv2.circle(frame,(int(point_list[index]),int(point_list[index+1])),5,color_code[color_id],-1)
					link_id = int(index/3)
					if link[link_id] != -1 and point_list[link[link_id]*3+2] !=0:
						frame = cv2.line(frame,(int(point_list[index]),int(point_list[index+1])),(int(point_list[link[link_id]*3]),int(point_list[link[link_id]*3+1])),color_code[color_id],3)
			color_id += 1
			color_id %= 9
		return frame, True
	return frame, False
				

	
def motion():
	# ------------------ #
	#  Video Read/Write  #
	# ------------------ #
	name = "video.avi"
	# Input reader
	cap = cv2.VideoCapture(name)
	fps = cap.get(cv2.CAP_PROP_FPS)
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

	# Output writer
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output1.avi', fourcc, fps, (w, h), True)

	# Read video frame by frame
	index = 1
	while True:
		# Get 1 frame
		success, frame = cap.read()

		if success:

			# Write 1 frame to output video
			outframe, result = process_img(index,frame)
			if result == False:
				break
			out.write(outframe)
			index += 1
		else:
			break

	# Release resource
	cap.release()
	out.release()

if __name__ == '__main__':
	motion()