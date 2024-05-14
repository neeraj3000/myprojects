import cv2
import numpy as np
from matplotlib import pyplot as plt


def region_of_intrest(img,vertices):
	black = np.zeros_like(img)
	# channel_count = img.shape[2]
	match_mask_color = 255 
	cv2.fillPoly(black,vertices,match_mask_color)
	mask_image = cv2.bitwise_and(img,black)
	return mask_image

def draw_the_lines(img,lines):
	blank = np.zeros_like(img)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(blank,(x1,y1),(x2,y2),(0,230,0),thickness=10)	
	img = cv2.addWeighted(img,0.8,blank,1,1)
	return img		

def process(img):
	# print(img.shape)
	height = img.shape[0]
	width = img.shape[1]

	region_of_intrest_vertices = [
		(0,height),

		(width/2,447),
		(width,height)
	]

	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	guass = cv2.GaussianBlur(gray,(5,5),0)
	canny_image = cv2.Canny(guass,100,120)
	new_img = region_of_intrest(canny_image,np.array([region_of_intrest_vertices],np.int32))
	lines = cv2.HoughLinesP(new_img,rho=2,theta=np.pi/180,threshold=85,lines=np.array([]),minLineLength=40,maxLineGap=250)
	final_img = draw_the_lines(img,lines)
	return final_img

cap = cv2.VideoCapture('/home/neeraj/Desktop/code/python/opencv/Advanced-Lane-Lines-master/project_video.mp4')
while(cap.isOpened()):
	ret,frame = cap.read()
	img = process(frame)
	cv2.imshow('lane_lines_detection',img)
	if cv2.waitKey(1)==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

