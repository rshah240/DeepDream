from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from deep_dream import DeepDream
import sys

try:
    path = str(sys.argv[1])
except IndexError:
    print('No Input given... please give a path to the image in the argument.')

img = image.load_img(path = path,target_size=(229,229,3))
img = image.img_to_array(img)
img = np.expand_dims(img,axis = 0)
img = preprocess_input(img)

inception_model = inception_v3.InceptionV3(weights = 'imagenet',include_top = False)
layer_settings = {
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}
dd = DeepDream(model = inception_model,layer_settings = layer_settings,
               input_image= img)

dd.deep_dream(result_prefix='output')
