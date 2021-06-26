import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



##ENCODER
latent_dim = 2

encoder_inputs = keras.Input(shape=(9, 1))
x = layers.Dense(7,  activation="relu",)(encoder_inputs)
x = layers.Dense(5,  activation="relu", )(x)
x = layers.Flatten()(x)
x = layers.Dense(2, activation="relu")(x)

#x  = tf.reshape(x,shape=(-1,1,16))
#x = layers.LSTM(16,name='memory_module')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()



##DECODE9
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(2, activation="relu")(latent_inputs)
#x = layers.Reshape((7, 7, 64))(x)
x = layers.Dense(5, activation="relu", )(x)
x = layers.Dense(7, activation="relu",)(x)
decoder_outputs = layers.Dense(9,  activation="sigmoid", )(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            data = data[0]
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)
            #print(data[0])
            #print(reconstruction)
            print(data[0])
            print(reconstruction[0])
            reconstruction_loss = keras.losses.mean_squared_error(data,reconstruction)
            '''reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )'''

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #print(kl_loss)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

##train
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

#(x_train, _), (x_test, _) = keras.datasets.imdb.load_data()
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32")

vae = VAE(encoder, decoder)


data = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/mappedDataNew.csv').values
trData = np.array(np.append(data[:,8].reshape(-1,1), np.asmatrix([data[:,10], data[:,11], data[:,12], data[:,22],
                                           data[:,1] ,data[:,26],data[:,27],data[:,15]]).T,axis=1)).astype(float)#data[:,26],data[:,27]

trData = np.nan_to_num(trData)
trData = np.array([k for k in trData if  str(k[0])!='nan' and  float(k[2])>=0 and float(k[4])>=0 and (float(k[3])>=8 ) and float(k[8])>0  ]).astype(float)

mnist_digitsTASKA = trData[:4000]
mnist_digitsTASKB = trData[4000:8000]


tasks = [mnist_digitsTASKA, mnist_digitsTASKB]
vae.compile(optimizer=keras.optimizers.Adam())
#vae.fit(mnist_digitsTASKB, epochs=30, batch_size=128)
#print(scipy.stats.ks_2samp(mnist_digitsTASKA, mnist_digitsTASKB))
for task in tasks:
    vae.fit(task, epochs=30, batch_size=128)
    print("END TASK----------------------\n\n")