
from Sample import Sample
import matplotlib.pyplot as plt
import numpy as np

class Sample_Classifier:
	def __init__(self,path,model):
		self.initialize_variables(path,model)
		self.classify()

	def initialize_variables(self,path,model):
		self.path = path
		self.genres = ['blues' , 'classical', 'country' , 'disco' , 'hiphop' , 
		'jazz' , 'metal' , 'pop' , 'reggae' , 'rock']
		self.sample = Sample(self.path)
		self.model = model
		self.genre_data = []
		self.predictions = []

	def classify(self,plot = False):
		for i in range(0,self.sample.image_number):
		  	self.data = self.sample.final_data[i].reshape((1,128,128,1))
		  	self.predictions.append(np.argmax(self.model.predict(self.data)))
		  	if max(self.model.predict(self.data)[0]) > 0.8:
		  		self.genre_data.append(
		  			np.argmax(self.model.predict(self.data)))
		try:
			self.probabilities = [i/self.sample.image_number 
				for i in np.bincount(self.predictions)]
			genre_index = np.argmax(np.bincount(self.genre_data))
		except: 
			genre_index = np.random.randint(10)
		self.genre = self.genres[genre_index]
		print('Genre is:',self.genre)