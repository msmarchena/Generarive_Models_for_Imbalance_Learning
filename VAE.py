
######################################
# Variational Autoencoders           #
# Author : Marlene Silva Marchena    #
######################################
'''This code is a based on
https://towardsdatascience.com/6-different-ways-of-implementing-vae-with-tensorflow-2-and-tensorflow-probability-9fe34a8ab981
'''
import numpy as np
import matplotlib.pyplot as plt

#from scipy.stats import norm


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

import tensorflow as tf
random_seed = 12345678
tf.random.set_seed(random_seed)

class VAE:
    
    def __init__(self):
        self.dim_x = 29
        self.intermediate_dim1 = 18
        self.intermediate_dim2 = 9
        self.dim_z = 3
        self.kl_weight = 1
        self.learning_rate = 0.05

    # build encoder model: q(z|X)
    def build_encoder(self):
        # define prior distribution for the code, which is an isotropic Gaussian
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.dim_z), scale=1.), 
                                reinterpreted_batch_ndims=1)
        # build layers argument for tfk.Sequential()
        input_shape = self.dim_x
        layers = [tfkl.InputLayer(input_shape=input_shape)]
        layers.append(tfkl.Dense(self.intermediate_dim1, activation='relu'))
        layers.append(tfkl.Dense(self.intermediate_dim2, activation='relu'))
        # the following two lines set the output to be a probabilistic distribution
        layers.append(tfkl.Dense(tfpl.IndependentNormal.params_size(self.dim_z), 
                                 activation=None, name='z_params'))
        layers.append(tfpl.IndependentNormal(self.dim_z, 
            convert_to_tensor_fn=tfd.Distribution.sample, 
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=self.kl_weight), 
            name='z_layer'))
        return tfk.Sequential(layers, name='encoder')

    # Sequential API decoder
    def build_decoder(self):
        layers = [tfkl.InputLayer(input_shape=self.dim_z)]
        layers.append(tfkl.Dense(self.intermediate_dim2, activation='relu'))
        layers.append(tfkl.Dense(self.intermediate_dim1, activation='relu'))
        layers.append(tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.dim_x)))      
        return tfk.Sequential(layers, name='decoder')

    def build_vae(self):
        x_input = tfk.Input(shape=self.dim_x)
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        z = encoder(x_input)

        # compile VAE model
        model = tfk.Model(inputs=x_input, outputs=decoder(z))

        model.compile(loss=negative_log_likelihood, 
                      optimizer=tfk.optimizers.Adam(self.learning_rate))
        return model

    def generate_data(self, n_samples, name="vae.h5"):
        # load model only for prediction (without training, for training use custom_objects={'my_custom_func': my_custom_func}
        # inside load_model)
        model = tfk.models.load_model(name, compile=False )
        # generate points in the latent space
        latent_points = np.random.normal(0, 1, (n_samples, self.dim_x ))
        latent_points = latent_points.reshape(n_samples, self.dim_x )
        X_vae = model.predict(latent_points)
        return X_vae    

# the negative of log-likelihood for probabilistic output
#negative_log_likelihood = tfkl.Lambda(lambda x, rv_x: -rv_x.log_prob(x))
negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)



