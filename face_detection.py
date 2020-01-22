import math
import numpy as np
import cv2

class face_detection( ) :

	def euclideanDist(self,a, b):
		return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))

	def find_eye_parameters(self,eye):
		#calculating eyes dimensions and center coordinates
		vertical_size=(self.euclideanDist(eye[1], eye[5])+self.euclideanDist(eye[2], eye[4]))/2
		horizental_size=self.euclideanDist(eye[0], eye[3])
		center_coor=[int((eye[0][1]+eye[3][1])/2) , int((eye[0][0]+eye[3][0])/2)]
		return (horizental_size/vertical_size),vertical_size,horizental_size,center_coor
	
	def find_mouth_parameters (self,mouth):
		#calculating mouth dimensions and center coordinates
		vertical_size=(self.euclideanDist(mouth[2], mouth[10])+self.euclideanDist(mouth[3], mouth[9])+self.euclideanDist(mouth[4], mouth[8]))/3
		horizental_size=self.euclideanDist(mouth[0], mouth[6])
		center_coor=[int((mouth[0][1]+mouth[6][1])/2) , int((mouth[0][0]+mouth[6][0])/2)]
		return (horizental_size/vertical_size),vertical_size,horizental_size,center_coor
	
	def image_cal(self,image):
		#image_calibration by cropping and resizing
		image=np.asarray(image)		
		image=image[100:380,140:500]
		image=cv2.resize(image,(480,640))	
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
	def predict( self,image ,sess,detection_graph) :
	
		#predicting face in the image
		bbox=None
		im_width,im_height,im_depth=image.shape
		image_np_expanded = np.expand_dims(image, axis=0)
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		scores = detection_graph.get_tensor_by_name('detection_scores:0')
		classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
		scores=np.squeeze(scores)
		boxes =np.squeeze(boxes)
		index_max = np.argmax(scores)
		if scores[index_max] > 0.5 :
			ymin, xmin, ymax, xmax=boxes[index_max]
			(left,  top, right,bottom) = (int(xmin * im_height+5), int(xmax * im_height-5),int(ymin * im_width+5 ), int(ymax * im_width -5))
			bbox=(left, right, top, bottom)
			return bbox	
		
