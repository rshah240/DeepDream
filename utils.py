import scipy
from tensorflow.keras.preprocessing import image
import numpy as np

def resize_img(img,size):
    img = np.copy(img)
    factors = (1,float(size[0])/img.shape[1], float(size[1])/img.shape[2],1)
    return scipy.ndimage.zoom(img,factors)

def save_img(img,fname):
    pil_img 