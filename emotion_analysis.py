import math
import pickle

class emotion_analyse ():
	def init(self):
		self.model=pickle.load(open('MLP_13cl.sav', 'rb'))
	
	def euclideanDist(self,a, b):
		return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))
	
	def analyse(self,outside_mouth,left_eyebrow,right_eyebrow,left_eye,right_eye,vertical_part_nose):
	
		f1=(self.euclideanDist(left_eye[1], left_eye[5])+self.euclideanDist(left_eye[2], left_eye[4]))/2
		f2=self.euclideanDist(left_eye[0], left_eye[3])
		f3=(self.euclideanDist(right_eye[1], right_eye[5])+self.euclideanDist(right_eye[2], right_eye[4]))/2
		f4=self.euclideanDist(right_eye[0], right_eye[3])	
		f5=(self.euclideanDist(outside_mouth[2], outside_mouth[10])+self.euclideanDist(outside_mouth[3], outside_mouth[9])+self.euclideanDist(outside_mouth[4], outside_mouth[8]))/3
		f6=self.euclideanDist(outside_mouth[0], outside_mouth[6])
		center_mouth=[int((outside_mouth[0][1]+outside_mouth[6][1])/2) , int((outside_mouth[0][0]+outside_mouth[6][0])/2)]
		f7=self.euclideanDist(center_mouth, vertical_part_nose[1])
		f8=self.euclideanDist(vertical_part_nose[1],left_eye[3])
		f9=self.euclideanDist(vertical_part_nose[1],left_eyebrow[2])
		f10=self.euclideanDist(vertical_part_nose[1],right_eye[3])
		f11=self.euclideanDist(vertical_part_nose[1],right_eyebrow[2])
		f12=self.euclideanDist(outside_mouth[0],left_eye[0])
		f13=self.euclideanDist(outside_mouth[0],left_eyebrow[0])
		f14=self.euclideanDist(outside_mouth[6],right_eye[0])
		f15=self.euclideanDist(outside_mouth[6],right_eyebrow[0])
		f16=self.euclideanDist(vertical_part_nose[1],right_eye[0])
		f17=self.euclideanDist(vertical_part_nose[1],right_eyebrow[0])
		f18=self.euclideanDist(vertical_part_nose[1],left_eye[0])
		f19=self.euclideanDist(vertical_part_nose[1],left_eyebrow[0])
		f20=self.euclideanDist(vertical_part_nose[1],left_eye[0])
		f21=self.euclideanDist(vertical_part_nose[1],left_eyebrow[0])
		f22=self.euclideanDist(right_eye[0],right_eyebrow[1])
		f23=self.euclideanDist(right_eye[3],right_eyebrow[1])
		f24=self.euclideanDist(left_eye[0],left_eyebrow[1])
		f25=self.euclideanDist(left_eye[3],left_eyebrow[1])
		f26=self.euclideanDist(center_mouth,right_eye[3])
		f27=self.euclideanDist(center_mouth,left_eye[0])

		features= [[f1,f2,f2,f3,f4,f5,f6,f7,f12,f14,f15,f24,f25]]
		prediction = self.model.predict(features) 

		if prediction==0 :
			state='neutral'
		elif prediction == 1 :
			state='happy'	
		else:
			state='unhappy'
		return state