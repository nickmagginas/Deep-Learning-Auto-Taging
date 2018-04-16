from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout,
 Activation, Flatten
from Dataset import Dataset
from keras.utils import plot_model
import numpy as np

class CNN:
	def __init__(self,test_acccuracy = False):
		self.initialize_variables()
		self.create_model()
		self.compile_and_train_model()
		if test_accuracy:
			self.calculate_accuracy()

	def initialize_variables(self):
		self.dataset = Dataset(create = False)
		print(len(self.dataset.train_x))

	def create_model(self):
		self.inp = Input(shape = (128,128,1))
		self.conv_1 = Convolution2D(32, (3,3),
		 padding='same', activation='relu')(self.inp)
		self.conv_2 = Convolution2D(64, (3,3),
		 padding='same', activation='relu')(self.conv_1)
		self.pool_1 = MaxPooling2D(pool_size=(2))(self.conv_2)
		self.drop_1 = Dropout(0.25)(self.pool_1)
		self.flat = Flatten()(self.drop_1)
		self.hidden = Dense(512, activation='relu')(self.flat)
		self.drop_3 = Dropout(0.5)(self.hidden)
		self.out = Dense(10, activation='softmax')(self.drop_3)
		self.model = Model(inputs=self.inp, outputs=self.out)

	def compile_and_train_model(self):
		self.model.compile(loss='categorical_crossentropy', optimizer='adam',
			 metrics=['accuracy'])
		self.dataset.train_x = np.expand_dims(self.dataset.train_x , axis = 4)
		self.model.fit(self.dataset.train_x, self.dataset.train_y,
			 batch_size=32, epochs=10, verbose=1,	validation_split = 0)
		try:
		    self.model.save(
		    	'C:\\Users\\nick\\Desktop\\Final_Project\\cnn_model.h5')
		    print('Model Saved')
		except Exception as e:
		    print('Model Save Failed. Error:' , e)

	def calculate_accuracy(self):
		_ , accuracy = self.model.evaluate(self.dataset.test_x, 
			self.dataset.test_y, verbose=1)
		print('Accuracy on test set:', 100*accuracy , '%')
		

if __name__ == '__main__':
	CNN()