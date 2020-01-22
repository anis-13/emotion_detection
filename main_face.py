
import dlib
import cv2
import os
import math
import tensorflow as tf
import numpy as np
from imutils import face_utils
import face_detection
import emotion_analysis

analyse_emotion=emotion_analysis.emotion_analyse()
face_det=face_detection.face_detection()
analyse_emotion.init()

#capturing images from USB
capture = cv2.VideoCapture(0)
capture.set(3,640)
capture.set(4,480)

#define indexes of keypoints on the face
right_eyebow_idxs=(2,7)
left_eyebrow_idxs=(7,12)
vert_nose=(12,16)
nose=[13,16,17,21]
left_eye_idxs =(21,27)
right_eye_idxs = (27,33)
mouth_idxs = (34,53)

#load the dlib facial landmark detector
predictor = dlib.shape_predictor('facial_trained_model.dat')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'Face_detection_model.pb'

# Label_map file
PATH_TO_LABELS = 'face_label_map.pbtxt'


#Tensorflow Model loading and configuration settings.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

#Declare some variables
state=None
idx=0      #index for frames counting.
measurement_begin = cv2.getTickCount()  #take time for 1minute blinks calculation


#flags For analysis
Flag_facial_ladmarks=False
Flag_face_direction=False
Flag_3D_Bbox= False
Flag_tiredness_analysis=True
Flag_emotion_analyse=False
emotion_freq=30   # set the value in frames for frequency of emeotion analyse

with detection_graph.as_default():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(graph=detection_graph, config=config) as sess:
		while True :
			ret, image_BGR = capture.read() 
					
			if ret :
				t_in = cv2.getTickCount()		
				#image calibration
				image=face_det.image_cal(image_BGR)
				#face localization 
				bbox=face_det.predict(image,sess,detection_graph)
				
				if bbox!= None :
				
					#get dlib prediction of selected features.
					bbox_face = dlib.rectangle(bbox[0]-20, bbox[1]-20, bbox[2]+20, bbox[3]+20)

					shape = face_utils.shape_to_np(predictor(image,bbox_face))
					
					#eyes shape selection
					left_eye = shape[left_eye_idxs[0]:left_eye_idxs[1]]					
					right_eye = shape[right_eye_idxs[0]:right_eye_idxs[1]]	
					
					#eyebrows shape selection
					left_eyebrow = shape[left_eyebrow_idxs[0]:left_eyebrow_idxs[1]]					
					right_eyebrow = shape[right_eyebow_idxs[0]:right_eyebow_idxs[1]]		
	
					#top and bottom of nose parts and full(xtop,ytop,xbottom,ybottom)
					vertical_part_nose=[shape[nose[0]][0],shape[nose[0]][1],shape[nose[1]][0],shape[nose[1]][1]]		
					horizental_part_nose=[shape[nose[2]][0],shape[nose[2]][1],shape[nose[3]][0],shape[nose[3]][1]]
					nose_full_shape=shape[vert_nose[0]:vert_nose[1]]

					#mouth points selection
					mouth=shape[mouth_idxs[0]:mouth_idxs[1]]
					outside_mouth=mouth[0:11,:]
					inside_mouth=mouth[12:21,:]

					#ANALYSE EMOTIONS
					if Flag_emotion_analyse:
						if idx % emotion_freq == 0 :
							state=analyse_emotion.analyse(outside_mouth,left_eyebrow,right_eyebrow,left_eye,right_eye,nose_full_shape)
						cv2.putText(image,'{}'.format(state),(240,bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), lineType=cv2.LINE_AA)
										
				
				# Calculate Frames per second (FPS)
				fps = round (cv2.getTickFrequency() / (cv2.getTickCount() - t_in) , 2)	
				cv2.putText(image,'FPS : {}  '.format(fps),(120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), lineType=cv2.LINE_AA) 
				cv2.imshow('Drive Face Analyse', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
				
				if(cv2.waitKey(1)==27):	break
        
			else :
				break
capture.release()
cv2.destroyAllWindows()
					

