######################################
# Generative Adversarial Network     #
# Author : Marlene Silva Marchena    #
######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import initializers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras.layers import Activation
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
import tensorflow as tf
random_seed = 12345678
tf.random.set_seed(random_seed)

class GAN:
    def __init__(self):
        self.n_classes = 2
        self.latent_dim = 29

        # Build the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # Build the combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = self.build_combine(self.generator, self.discriminator)

    def build_generator(self):

        inputs = Input(shape=(self.latent_dim,))

        l1 = Dense(128, input_dim=self.latent_dim,kernel_initializer=initializers.glorot_normal(seed=random_seed))(inputs)

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

    def build_discriminator(self):

        inputs = Input(shape=(self.latent_dim,))

        l1 = Dense(128, input_dim=self.latent_dim,kernel_initializer=initializers.glorot_normal(seed=random_seed))(inputs)
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
        outputs = Activation("sigmoid")(outputs)

        model = Model(inputs=inputs, outputs=outputs)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        return model

    def build_combine(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False

        gan_input = Input(shape=(self.latent_dim,))
        gan_output = discriminator(generator(gan_input))
        model = Model(gan_input, gan_output)

        # compile model. It uses the same loss and opt as the discriminator model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        return model        

    def train(self, dataset, epochs=100, batch_size=100, name="gan"):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # lists for keeping track of loss
        d_hist, g_hist = list(), list()

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of real data
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            X_real = dataset[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of fake data
            X_fake = self.generator.predict(noise)

            # update discriminator model weights
            d_loss_real, _ = self.discriminator.train_on_batch(X_real, valid)
            d_loss_fake, _ = self.discriminator.train_on_batch(X_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_hist.append(d_loss)

            # ---------------------
            #  Train Generator
            # ---------------------

            # update the generator via the discriminator's error, with inverted labels
            g_loss, _ = self.combined.train_on_batch(noise, valid)
            g_hist.append(g_loss)

            # summarize loss on this batch
            print(
                ">%d, d1=%.3f, d2=%.3f, d=%.3f, g=%.3f"
                % (epoch + 1, d_loss_real, d_loss_fake, d_loss, g_loss)
            )
        # line plots of loss
        plt.plot(d_hist, label="Discriminator", color="red")
        plt.plot(g_hist, label="Generator", color="blue")
        plt.title(f'Training Losses - {name}')
        plt.legend()
        plt.savefig(f'{name}.png')
        plt.show()
        plt.close()

        # save the generator model
        self.generator.save(f'{name}.h5')


    def generate_data(self, n_samples, name="gan.h5"):
        # load model
        model = load_model(name)
        # generate points in the latent space
        latent_points = np.random.normal(0, 1, (n_samples, self.latent_dim))
        latent_points = latent_points.reshape(n_samples, self.latent_dim)
        X_gan = model.predict(latent_points)
        return X_gan