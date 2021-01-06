######################################
# Wasserstein GAN                    #
# Author : Marlene Silva Marchena    #
######################################
# This code is adapted to be used with multiprocessing
# Net configuration is changed to 400-800
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, LeakyReLU , Dropout
from keras.layers import Activation #, Flatten

from keras.utils import plot_model

from keras import backend
from keras.constraints import Constraint

from keras.initializers import RandomNormal
from keras.optimizers import RMSprop

import tensorflow as tf
random_seed = 12345678
tf.random.set_seed(random_seed)

class WGAN:
    def __init__(self):
        self.n_classes = 2
        self.latent_dim = 29
        self.n_critic = 5
        self.clip_value = 0.01

        # Build critic
        self.critic = self.build_critic()

        # Build the generator
        self.generator = self.build_generator()

        # Build the combined model  (stacked generator and critic)
        self.combined = self.build_combine(self.generator, self.critic)

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def build_generator(self):

        inputs = Input(shape=(self.latent_dim,))

        l1 = Dense(128, input_dim=self.latent_dim)(inputs)
        l1 = LeakyReLU(0.2)(l1)

        l2 = Dense(256)(l1)
        l2 = LeakyReLU(0.2)(l2)

        l3 = Dense(128)(l2)
        l3 = LeakyReLU(0.2)(l3)

        l4 = Dense(80)(l3)
        l4 = LeakyReLU(0.2)(l4)

        output = Dense(self.latent_dim)(l4)
        output = Activation("tanh")(output)

        # This model is not compiled
        model = Model(inputs=inputs, outputs=output)

        return model

    def build_critic(self):

        inputs = Input(shape=(self.latent_dim,))

        l1 = Dense(128, input_dim=self.latent_dim)(inputs)
        l1 = LeakyReLU(0.2)(l1)
        l1 = Dropout(0.4)(l1)

        l2 = Dense(256)(l1)
        l2 = LeakyReLU(0.2)(l2)
        l2 = Dropout(0.4)(l2)

        l3 = Dense(128)(l2)
        l3 = LeakyReLU(0.2)(l3)
        l3 = Dropout(0.4)(l3)

        l4 = Dense(80)(l3)
        l4 = LeakyReLU(0.2)(l4)
        l4 = Dropout(0.4)(l4)

        outputs = Dense(1)(l4)

        model = Model(inputs=inputs, outputs=outputs)

        # compile model
        opt = RMSprop(lr=0.0005)
        model.compile(loss=self.wasserstein_loss, optimizer=opt, metrics=["accuracy"])

        return model

    def build_combine(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False

        wgan_input = Input(shape=(self.latent_dim,))
        wgan_output = discriminator(generator(wgan_input))
        model = Model(wgan_input, wgan_output)

        # compile model. It uses the same loss and opt as the discriminator model
        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=opt, metrics=["accuracy"])

        return model   

    def train(self, dataset, epochs=100, batch_size=100, name="wgan"):
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        # lists for keeping track of loss
        c1_hist, c2_hist, g_hist = list(), list(), list()

        for epoch in range(epochs):
            # update the critic more than the generator
            c1_tmp, c2_tmp = list(), list()
            for _ in range(self.n_critic):

                #  Train Discriminator

                # Select a random batch of images
                idx = np.random.randint(0, dataset.shape[0], batch_size)
                X_real = dataset[idx]
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                X_fake = self.generator.predict(noise)

                # Train the critic

                d_loss_real, _ = self.critic.train_on_batch(X_real, valid)
                d_loss_fake, _ = self.critic.train_on_batch(X_fake, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                c1_tmp.append(d_loss_real)
                c2_tmp.append(d_loss_fake)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # store critic loss
            c1_hist.append(np.mean(c1_tmp))
            c2_hist.append(np.mean(c2_tmp))
            #  Train Generator

            # Sample noise as generator input
            g_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_valid = -np.ones((batch_size, 1))

            # update the generator via the discriminator's error, with inverted labels
            g_loss, _ = self.combined.train_on_batch(g_noise, g_valid)
            g_hist.append(g_loss)

            # summarize loss on this batch
            print(
                ">%d, c1=%.3f, c2=%.3f, g=%.3f"
                % (epoch + 1, d_loss_real, d_loss_fake, g_loss)
            )

        # line plots of loss
        plt.plot(c1_hist, label="Critic_real", color="green")
        plt.plot(c2_hist, label="Critic_fake", color="red")
        plt.plot(g_hist, label="Generator", color="blue")
        plt.title(f'Training Losses - {name}')
        plt.legend()
        plt.savefig(f'{name}.png')
        plt.show()
        plt.close()

        self.generator.save(f'{name}.h5')

    def generate_data(self, n_samples, name="wgan.h5"):
        # load model
        model = load_model(name)
        # generate points in the latent space
        latent_points = np.random.normal(0, 1, (n_samples, self.latent_dim))
        latent_points = latent_points.reshape(n_samples, self.latent_dim)
        X_gan = model.predict(latent_points)
        return X_gan
