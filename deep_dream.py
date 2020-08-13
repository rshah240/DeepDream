import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import hyperparameters
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class DeepDream:
    def __init__(self,model,layer_settings,input_image):
        """Initialising the model"""
        self.input_image = input_image
        self.layer_settings = layer_settings
        outputs_dict = dict(
            [
                (layer.name, layer.output)
                for layer in [model.get_layer(name) for name in self.layer_settings.keys()]
            ]
        )
        self.feature_extractor = Model(inputs = model.input, outputs = outputs_dict)


    def compute_loss(self,input_image):
        features = self.feature_extractor(input_image)
        #Initialize the loss
        loss = tf.zeros(shape=())
        for name in features.keys():
            coeff = self.layer_settings[name]
            activation = features[name]
            #We avoid border artifacts by only involving non border pixels in the loss
            scaling = tf.reduce_prod(tf.cast(tf.shape(activation),tf.float32))
            loss += coeff* tf.reduce_sum(tf.square(activation[:,2:-2,2:-2,:])) / scaling
        return loss

    @tf.function
    def gradient_ascent_step(self,input_image):
        with tf.GradientTape() as g:
            g.watch(input_image)
            loss = self.compute_loss(input_image)
        #Compute Gradients
        grads = g.gradient(loss,input_image)
        #Normalize Gradients
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)),1e-6)
        input_image += hyperparameters.step * grads
        return loss,input_image

    def gradient_ascent_loop(self,input_image):
        iterations = hyperparameters.iterations
        for i in range(iterations):
            loss,input_image = self.gradient_ascent_step(input_image)
            if hyperparameters.max_loss is not None and loss > hyperparameters.max_loss:
                break
            print(".... Loss Value at Step {} is {}".format(i,loss))
            return input_image

    @staticmethod
    def deprocess_image(x):
        """utility function to convert a float array into a valid uint8 image
        #Arguments
            x: a numpy array representing the generated image
        #:returns
            A processed numpy array, which could be used in e.g. imshow
        """
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.25

        x += 0.5
        x = np.clip(x, 0, 1)

        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def deep_dream(self,result_prefix):
        original_shape = self.input_image.shape[1:3]
        successive_shapes = [original_shape]
        num_octaves = hyperparameters.num_octaves
        for i in range(1,num_octaves):
            shape = tuple([int(dim /(hyperparameters.octave_scale ** i)) for dim in original_shape ])
            successive_shapes.append(shape)
        successive_shapes = successive_shapes[::-1] #Reversing the list
        shrunk_original_img = tf.image.resize(self.input_image,successive_shapes[0])

        img = tf.identity(self.input_image) #Make a copy
        for i,shape in enumerate(successive_shapes):
            print("Processing Octave {} with shape {}".format(i,shape))
            img = tf.image.resize(img,shape)
            img = self.gradient_ascent_loop(img)

            upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img,shape)
            same_size_original = tf.image.resize(self.input_image, shape)
            lost_detail = same_size_original - upscaled_shrunk_original_img

            img += lost_detail
            shrunk_original_img = tf.image.resize(self.input_image,shape)
        image.save_img(result_prefix + ".png",DeepDream.deprocess_image(img[0].numpy()))














