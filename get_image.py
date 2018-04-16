from keras.models import load_model
from keras.utils import plot_model
model = load_model('C:\\Users\\nick\\Desktop\\Final_Project\\test_cnn_model.h5')
plot_model(model, to_file = 'cnn.png', show_shapes = True, show_layer_names = True)