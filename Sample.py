import soundfile as sf
import numpy as np
from scipy.fftpack import fft
from warnings import warn
import matplotlib.pyplot as plt
import sys
import time

class Sample:
    def __init__(self,path):
        self.initialize_variables(path)
        self.audioread()
        self.create_images()
    
    def initialize_variables(self,path):
        self.path = path
        self.image_height = 128
        self.image_width = 128
        self.time_per_image = 2
        self.final_data = []
        self.print_audioread = False
        self.plot_samples = False
        self.index = 0 
      
    def audioread(self):
        try :
            self.data , self.fs = sf.read(self.path)
            if self.print_audioread:
                print(self.fs)
                print('Audioread Successful')
        except Exception as e:
            print('Invalid Type of Input Audio. Error:' , e , '\n. Exiting...')
            sys.exit(0)

        if isinstance(self.data[0],np.ndarray):
            warn('Warning: Audio is Stereo. Only Keeping One Channel', DeprecationWarning)
            self.data = list(map(lambda x : x[1] , self.data))
        self.song_length = len(self.data)/self.fs
        self.fs = 22050
        

    def create_images(self):
        self.image_number = int(self.song_length/self.time_per_image)
        self.dimension = self.image_number*self.image_width
        self.L = len(self.data)
        for i in np.linspace(0,1-1/self.dimension,self.dimension):
            segment = self.data[int(i*self.L):int((i+1/self.dimension)*self.L)] 
            self.frequency_transform(segment)
            self.quantize()
            self.final_data.append(self.discrete_amplitudes)
        self.final_data = np.array(self.final_data)
        self.final_data = np.array(np.split(self.final_data,self.image_number))
            
    def frequency_transform(self,time_data):
        self.frequency_data = np.abs(fft(time_data))[:int(len(time_data)/2)]
        if self.fs == 44100:
            warn('Large Sampling Rate. Reducing Dimensionality', DeprecationWarning)
            self.frequency_data = self.frequency_data[:int(len(self.frequency_data)/2)]
            self.fs = 22050
        self.frequencies = np.linspace(0,self.fs/2,len(self.frequency_data))
        if self.plot_samples and self.index < 100 : 
            print(len(self.frequencies))
            plt.plot(self.frequencies,self.frequency_data)
            plt.xlabel('Frequency(Hz)')
            plt.ylabel('Amplitude')
            plt.title('Frequency Domain')
            plt.show()
            self.index += 1 
        if all(i for i in self.frequency_data) == 0:
            warn('Warning: Frequency Information Is Zero', DeprecationWarning)
            
    def quantize(self):
        self.discrete_amplitudes = np.zeros(self.image_height)
        self.steps = [i*2 for i in np.linspace(1,self.fs/4,self.image_height-1)]
        self.discrete_frequency_levels = np.digitize(self.frequencies,self.steps)
        for index , level in enumerate(self.discrete_frequency_levels):
            self.discrete_amplitudes[level] += self.frequency_data[index]
