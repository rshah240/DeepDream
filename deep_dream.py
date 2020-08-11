import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class DeepDream:
    def __init__(self,model,layer_contributions,input_image):
        """Initialising the model"""
        self.model = model
        self.layer_contributions = layer_contributions
        self.input_image = input_image

    @tf.function
    def get_loss(self):
        layer_dict = dict([(layer.name,layer) for layer in self.model.layers])
        loss = tf.Variable(0.)
        for layer_name in self.layer_contributions:
            coefficient =  self.layer_contributions[layer_name]
            activation = layer_dict[layer_name].output
            scaling = tf.reduce_prod(tf.cast(tf.shape(activation),tf.float32))
            loss += coefficient * tf.reduce_sum(tf.square(activation[:,2:-2,2:-2,:]))/scaling
        return loss
    @tf.function
    def gradient_ascent(self,iterations,max_loss,step):
        dream = self.model.input
        with tf.GradientTape() as g:
            g.watch(dream)
            loss =  self.get_loss()
        grads = g.gradient(loss,self.input_image)[0]
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)),1e-7)
        fetch_loss_and_grads = K.function([dream],[grads,loss])
        for i in range(iterations):
            grads,loss = fetch_loss_and_grads(self.input_image)
            if max_loss is not None and loss > max_loss:
                break
            print('Loss Value at {} : {}'.format(i,loss))
            self.input_image += step*grads
        return self.input_image



