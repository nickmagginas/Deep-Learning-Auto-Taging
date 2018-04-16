from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from Classify_Sample import Sample_Classifier
import numpy as np

class GUI:
	def __init__(self,master):
		self.master = master
		self.master.title('Audio Classifier')
		self.master.geometry('1000x800')
		self.master.resizable(0, 0)
		self.start_called = False
		self.end_called = False
		self.genre_called = False
		self.genres = ['blues' , 'classical', 'country' , 'disco' , 'hiphop' , 'jazz' , 'metal' , 'pop' , 'reggae' , 'rock']
		self.master.title('Audio Classifier')
		self.path = None
		self.genre = None
		self.does_not = Label(self.master , text = 'Cannot Load File')
		self.application()


	def application(self):
		if not self.path and not self.start_called:
			self.label = Label(self.master , text = 'Browse to find File')
			self.label.pack()

			Button(self.master, text = "Browse", command = self.loadtemplate, width = 10).pack()

			self.start_called = True

		if self.genre and not self.genre_called:
			self.genre_label = Label(self.master, text = 'Genre is: ' + self.genre)
			self.genre_label.pack() 
			self.genre_called = True
			self.fig = Figure(figsize = (12,6))
			self.a = self.fig.add_subplot(111)
			while len(self.probabilities) != 10: 
				self.probabilities.append(0)
			self.a.bar(np.linspace(0,10,10) ,self.probabilities)
			self.a.set_xticks(np.linspace(0,10,10))
			self.a.set_xticklabels(self.genres)
			self.a.set_ylabel('Probabilities(%)', fontsize=14)
			self.a.set_xlabel('Genre', fontsize=14)
			self.a.set_title('Probabilites per Genre')
			self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
			self.canvas.get_tk_widget().pack()
			self.canvas.draw()

	def loadtemplate(self): 
		filename = filedialog.askopenfilename(filetypes = (("Wav Files", "*.wav")
		                                                     ,("All files", "*.*") ))
		self.path = filename
		self.classify()
	
	def classify(self):
		self.does_not.pack_forget()
		try:
			sample = Sample_Classifier(self.path)
			self.genre = sample.genre
			self.probabilities = sample.probabilities
		except: 
			if len(self.master.pack_slaves()) == 2:
				self.does_not.pack()
		self.application()


root = Tk()
gui = GUI(root)
root.mainloop()
