from Classify_Sample import Sample_Classifier
from keras.models import load_model
import numpy as np

def main():
	genres =  ['blues' , 'classical', 'country' , 'disco' , 'hiphop' , 'jazz' ,
	 'metal' , 'pop' , 'reggae' , 'rock']
	main_path = 'C:\\Users\\nick\\Desktop\\Audio_Classification\\Genres'
	actual_genres = []
	predict_genres = []
	model = load_model(
		'C:\\Users\\nick\\Desktop\\Final_Project\\cnn_model.h5')
	indices = np.load(
		'C:\\Users\\nick\\Desktop\\Final_Project\\excluded_indices.npy')
	print(indices)
	for index,i in enumerate(genres):
		for x in indices[index]:
			counts = 0
			print(main_path+'\\'+i+'\\'+i+'.'+'%05d'%x+'.au')
			sample = Sample_Classifier(
				main_path+'\\'+i+'\\'+i+'.'+'%05d'%x+'.au',model)
			predict_genres.append(sample.genre)
			actual_genres.append(i)
			for index,i in enumerate(actual_genres):
				if i == predict_genres[index]:
					counts += 1
			print('Accuracy:', 100*counts/len(actual_genres), '%')

	print(predict_genres)
	print(actual_genres)

if __name__ == '__main__': 
	main()