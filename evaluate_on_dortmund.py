from keras.models import load_model
import os
import sys
from Sample import Sample
import numpy as np


class Evaluation:
	def __init__(self):
		self.initialize_model()
		self.predict_dataset()

	def initialize_model(self):
		self.model = load_model('C:\\Users\\nick\\Desktop\\Final_Project\\test_cnn_model.h5')
		self.genres = ['blues' , 'classical', 'country' , 'disco' , 'hiphop' , 'jazz' , 'metal' , 'pop' , 'reggae' , 'rock']

	def get_number_of_samples(self,genre):
		try:
			return len(os.listdir('C:\\Users\\nick\\Desktop\\Dortmund_Genres\\Dataset\\'+genre)), 'C:\\Users\\nick\\Desktop\\Dortmund_Genres\\Dataset\\'+genre
		except Exception as e:
			print('Cannot List Directory. Error:',e)
			print('Exiting...')
			sys.exit(0)

	def classify(self,sample):
		genre_data = []
		for i in range(0,sample.image_number):
			data = sample.final_data[i].reshape((1,128,128,1))
			prediction = self.model.predict(data)[0]
			if max(prediction) > 0.6:
				genre_data.append(np.argmax(prediction))
		try:	
			genre_index = np.argmax(np.bincount(genre_data))
		except Exception as e: 
					print('Exception:', e)
					genre_index = np.random.randint(9)
		return self.genres[genre_index]



	def predict_dataset(self):
		genres = ['blues','folkcountry','jazz','pop','rock']
		n_samples = 0
		counts = 0
		for genre in genres:
			new_n_samples , genre_path = self.get_number_of_samples(genre)
			n_samples += new_n_samples
			for i in range(0,new_n_samples):
				sample = Sample(genre_path + '\\' + genre + '.' + str(i).zfill(3) + '.wav')
				predict_genre = self.classify(sample)
				if genre != predict_genre:
					if (predict_genre == 'country' and genre == 'folkcountry') or (predict_genre == 'hiphop' and genre == 'raphiphop') or predict_genre == None:
						pass
					else:
						counts += 1
			print(counts/n_samples) 

Evaluation()