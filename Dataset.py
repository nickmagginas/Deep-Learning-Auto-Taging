import time
import random
from sklearn.model_selection import train_test_split
from Sample import Sample
import numpy as np

class Dataset:
    def __init__(self,create = False): 
        self.initialize_variables()
        if not self.load_from_files() or create:
            self.get_data()
            self.split_data()
            self.save_data()
        print('Execution Time:' , time.time() - self.start_time, 's')
    
    def initialize_variables(self):
        self.start_time = time.time()
        self.genres = ['blues' , 'classical', 'country' , 'disco' , 'hiphop' 
        , 'jazz' , 'metal' , 'pop' , 'reggae' , 'rock']
        self.main_path = 'C:\\Users\\nick\\Desktop\\Audio_Classification\\Genres'
        self.n_samples = 80
        self.excluded = []
        self.audio_data = []
        self.labels = []

    def load_from_files(self):
        try:
            self.train_x = np.load(
                'C:\\Users\\nick\\Desktop\\Final_Project\\train_x_data.npy')
            self.train_y = np.load(
                'C:\\Users\\nick\\Desktop\\Final_Project\\train_y_data.npy')
            self.test_x = np.load(
                'C:\\Users\\nick\\Desktop\\Final_Project\\test_x_data.npy')
            self.test_y = np.load(
                'C:\\Users\\nick\\Desktop\\Final_Project\\test_y_data.npy')
            return True
        except Exception as e: 
            print('Cannot import files. Error:',e)
            print('Reconstructing Dataset')
            return False

    
    def get_data(self):
        print('Beggining audio file reading...')
        for x in range(0 , len(self.genres)): 
            genre_path = self.main_path + '\\' + self.genres[x]
            drop_indices = sorted(random.sample(range(1,100),20))
            self.excluded.append(drop_indices)
            sample_list = list(range(0,100))
            for i in reversed(drop_indices): 
                del sample_list[i]
            for i in range(0,self.n_samples):
                current_count = '.' + '%05d' % i
                song_path = genre_path + '\\' + self.genres[x] + current_count + '.au'
                sample = Sample(song_path)
                self.audio_data.extend(sample.final_data)
                for y in range(0,sample.image_number):  
                    self.labels.append(x)
                if i%10 == 0:
                    print('Percentage Complete :' , 
                        int(((x*100)+i)/(len(self.genres)*100)*100) , '%')
        print('Audio read succesfully')
        print('Saving droped indices')
        self.excluded = np.array(self.excluded)
        print(self.excluded)
        try: 
            np.save('C:\\Users\\nick\\Desktop\\Final_Project\\excluded_indices',
                self.excluded)
        except Exception as e: 
            print('Write Failed')
        self.audio_data = np.array(self.audio_data)
        self.labels = np.array(self.labels)
        self.labels_onehot = (np.arange(len(self.genres)) == 
            self.labels[: , None]).astype(int)
        
    def save_data(self):
        try:
            print('Writing arrays to file...')
            np.save('C:\\Users\\nick\\Desktop\\Final_Project\\train_x_data',
                self.train_x)
            np.save('C:\\Users\\nick\\Desktop\\Final_Project\\train_y_data',
                self.train_y)
            np.save('C:\\Users\\nick\\Desktop\\Final_Project\\test_x_data',
                self.test_x)
            np.save('C:\\Users\\nick\\Desktop\\Final_Project\\test_y_data',
                self.test_y)
            print('Write Complete')
        except Exception as e:
            print('Write Failed')
            print('Error:', e)
        
    def split_data(self):
        self.train_x , self.test_x, 
        self.train_y, self.test_y = train_test_split(self.audio_data,
            self.labels_onehot,test_size = 0.00)

dataset = Dataset(create = True)