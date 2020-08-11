import scipy
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import numpy as np

def resize_img(img,size):
    img = np.copy(img)
    factors = (1,float(size[0])/img.shape[1], float(size[1])/img.shape[2],1)
    return scipy.ndimage.zoom(img,factors)

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3,x.shape[2],x.shape[1]))
        x = x.transpose((1,2,0))
    else:
        x = x.reshape((x.shape[1],x.shape[2],3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x,0,255).astype('uint8')
    return x
def save_img(img,fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname,pil_img)

