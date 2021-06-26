import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# lstm autoencoder recreate sequence
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
import numpy as np
import tensorflow.keras.backend as K
import matplotlib as plt
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
from AttentionDecoder import AttentionDecoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from random import randrange, sample
from pyinform import conditional_entropy
from  tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

##Entropy
from skgof import ks_test, cvm_test, ad_test
def getData():
    data = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/mappedDataNew.csv').values
    draft = data[:,8].reshape(-1,1)
    wa = data[:,10]
    ws = data[:,11]
    stw = data[:,12]
    swh= data[:,22]
    bearing = data[:,1]
    lat = data[:,26]
    lon = data[:,27]
    foc = data[:,15]

    trData = np.array(np.append(draft, np.asmatrix([wa, ws, stw, swh,
                                              bearing ,foc]).T,axis=1)).astype(float)#data[:,26],data[:,27]

    trData = np.nan_to_num(trData)
    trData = np.array([k for k in trData if  str(k[0])!='nan' and  float(k[2])>0 and float(k[4])>0 and (float(k[3])>=8 ) and float(k[6])>0  ]).astype(float)


    return trData

# reshape input into [samples, timesteps, features]
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        #seq_x, seq_y = sequence[i:end_ix][:, 0:sequence.shape[1] - 1], sequence[end_ix - 1][sequence.shape[1] - 1]
        seq_ = sequence[i:end_ix][:, :]
        X.append(seq_)
        #y.append(seq_y)
    return array(X)

    # define input sequence

trData = getData()

raw_seq = trData#[:4000]
seqLSTM = raw_seq
# split into samples
#seqLSTM = split_sequence(raw_seq, n_steps)
#seqLSTM = seqLSTM.reshape(-1,n_steps,9)
n_steps = 1
dataLength = len(seqLSTM)
seqLSTMA = seqLSTM[14000:19000]

seqLSTMAmem = split_sequence(seqLSTMA,n_steps)
seqLSTMAmem = seqLSTMAmem.reshape(-1,n_steps,7)

seqLSTMB = seqLSTM[30000:35000]

plt.show()
#scipy.stats.ks_2samp(seqLSTMA, seqLSTMB)
# call MinMaxScaler object
min_max_scaler = MinMaxScaler()
X_train_normA = min_max_scaler.fit_transform(seqLSTMA)
X_train_normB = min_max_scaler.fit_transform(seqLSTMB)

tasks = [X_train_normA.reshape(len(seqLSTMA),n_steps,7), X_train_normB.reshape(len(seqLSTMB),n_steps,7)]

def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a time series."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)\
    +tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)


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
            #data = data[0]
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)
            #print(data[0])
            #print(reconstruction)
            #print(data[0])
            #print(reconstruction[0])
            reconstruction_loss =  keras.losses.mean_squared_error(data, reconstruction)
                #keras.losses.mean_squared_error(data,reconstruction)
            '''reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.sparse_categorical_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )'''

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #print(kl_loss)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss #+ kl_loss
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

class LSTM_AE_IW(keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(LSTM_AE_IW, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        #self.seqlstma = seqlstma
        #self.seqlstmb = seqlstmb
        #self.mem = mem
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker

        ]

    def train_step(self, data):

        with tf.GradientTape() as tape:
            #data = data[0]

            encoderOutput = self.encoder(data)
            reconstruction = self.decoder(encoderOutput)

            reconstruction_loss = keras.losses.mean_squared_error(data,reconstruction) #+ keras.losses.kullback_leibler_divergence(encoderOutput, reconstruction)

            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        #self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),

        }

class LSTM_AE(keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(LSTM_AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        #self.seqlstma = seqlstma
        #self.seqlstmb = seqlstmb
        #self.mem = mem
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")



    @property
    def metrics(self):
        return [
            self.total_loss_tracker

        ]

    def train_step(self, data):

        with tf.GradientTape() as tape:
            #data = data[0]

            encoderOutput = self.encoder(data)
            reconstruction = self.decoder(encoderOutput)

            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction) \
                                  #+  keras.losses.kullback_leibler_divergence(data,reconstruction)
            #

            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        #self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),

        }


def vae_LSTM_Model():
    ##ENCODER
    latent_dim = 100

    encoder_inputs = keras.Input(shape=(n_steps,1000))
    x = layers.LSTM(500, return_sequences=True )(encoder_inputs)
    #x = layers.Dense(200, )(x)
    #x = layers.Flatten()(x)
    x = TimeDistributed(layers.Dense(1000,  ))(x)

    # x  = tf.reshape(x,shape=(-1,1,16))
    # x = layers.LSTM(16,name='memory_module')(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    ##DECODER
    latent_inputs = keras.Input(shape=(n_steps, latent_dim))
    x = layers.LSTM(1000,return_sequences=True )(latent_inputs)
    # x = layers.Reshape((7, 7, 64))(x)
    #x = layers.Dense(200,  )(x)
    x = layers.LSTM(500, return_sequences=True )(x)
    decoder_outputs = TimeDistributed(layers.Dense(1000,  ))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return encoder, decoder

def VAE_windwowIdentificationModel():
    ##ENCODER
    latent_dim = 100
    input_dim = 1000

    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(input_dim, )(encoder_inputs)
    x = layers.Dense(500, )(x)
    #x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, )(x)

    # x  = tf.reshape(x,shape=(-1,1,16))
    # x = layers.LSTM(16,name='memory_module')(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    ##DECODE9
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(latent_dim,)(latent_inputs)
    # x = layers.Reshape((7, 7, 64))(x)
    x = layers.Dense(500,  )(x)
    #x = layers.Dense(700, activation="relu", )(x)
    decoder_outputs = layers.Dense(input_dim,  )(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return encoder, decoder


def windwowIdentificationModelALT():
    timesteps = 1 # Length of your sequences
    input_dim = 1000
    latent_dim = 1000
    features = 7
    output_dim = 2000
    inputs = keras.Input(shape=(n_steps,7 ))

    #encoded = layers.LSTM(700,kernel_initializer='he_uniform',return_sequences=True,activation='relu')(inputs)
    #encoded = layers.Dense(input_dim, activation='relu')(inputs)

    encoded = layers.LSTM(3,return_sequences=True,)(inputs)
    #encoded = layers.LSTM(1500, return_sequences=True, )(encoded)
    #encoded = layers.LSTM(900, return_sequences=True, )(encoded)


    encoded = TimeDistributed(layers.Dense(7,  ))(encoded)
        #TimeDistributed(layers.Dense(latent_dim, ))(encoded)
        #layers.LSTM(latent_dim, return_sequences=True )(encoded)#
        #

    #encoded = layers.Dense(latent_dim, activation='relu')(encoded)

    #encoded = layers.RepeatVector(features)(encoded)

    latent_inputs = keras.Input(shape=( n_steps, 7,))


    decoded = layers.LSTM(3, return_sequences=True,name='dec2',)(latent_inputs)
    #decoded = layers.LSTM(1500, return_sequences=True, name='dec3',)(decoded)
    #decoded = layers.LSTM(700, return_sequences=True, name='dec3')(decoded)
    #decoded = layers.LSTM(900, return_sequences=True, name='dec4')(decoded)
    #decoded = layers.LSTM(700, kernel_initializer='he_uniform', activation='relu', return_sequences=True, name='dec3')(decoded)
    decoded = TimeDistributed(layers.Dense(features, ))(decoded)
    #decoded = layers.Dense(input_dim, activation="relu", )(decoded)

    #sequence_autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)
    decoder = keras.Model(latent_inputs, decoded, name="decoder")

    return  encoder , decoder

def windwowIdentificationModelGEN():
    timesteps = 1 # Length of your sequences
    input_dim = 2000
    latent_dim = 2000
    features = 7
    output_dim = 2000
    inputs = keras.Input(shape=(n_steps,input_dim ))

    #encoded = layers.LSTM(700,kernel_initializer='he_uniform',return_sequences=True,activation='relu')(inputs)
    #encoded = layers.Dense(input_dim, activation='relu')(inputs)

    encoded = layers.LSTM(1000,return_sequences=True,)(inputs)
    #encoded = layers.LSTM(1500, return_sequences=True, )(encoded)
    #encoded = layers.LSTM(900, return_sequences=True, )(encoded)


    encoded = TimeDistributed(layers.Dense(latent_dim,  ))(encoded)
        #TimeDistributed(layers.Dense(latent_dim, ))(encoded)
        #layers.LSTM(latent_dim, return_sequences=True )(encoded)#
        #

    #encoded = layers.Dense(latent_dim, activation='relu')(encoded)

    #encoded = layers.RepeatVector(features)(encoded)

    latent_inputs = keras.Input(shape=( n_steps,latent_dim,))


    decoded = layers.LSTM(1000, return_sequences=True,name='dec2',)(latent_inputs)
    #decoded = layers.LSTM(1500, return_sequences=True, name='dec3',)(decoded)
    #decoded = layers.LSTM(700, return_sequences=True, name='dec3')(decoded)
    #decoded = layers.LSTM(900, return_sequences=True, name='dec4')(decoded)
    #decoded = layers.LSTM(700, kernel_initializer='he_uniform', activation='relu', return_sequences=True, name='dec3')(decoded)
    decoded = TimeDistributed(layers.Dense(input_dim, ))(decoded)
    #decoded = layers.Dense(input_dim, activation="relu", )(decoded)

    #sequence_autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)
    decoder = keras.Model(latent_inputs, decoded, name="decoder")

    return  encoder , decoder

def lstmAUTO():

    timesteps = 1000
    input_dim = 7
    latent_dim = 1000

    inputs = keras.Input(shape=( 1000,input_dim))
    encoded = layers.LSTM(500, return_sequences=True)(inputs)
    encoded = layers.LSTM(1000, return_sequences=True)(encoded)

    #decoded = layers.RepeatVector(input_dim)(encoded)

    decoded = layers.LSTM(500, return_sequences=True)(encoded)
    decoded = layers.LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)
    #decoder = keras.Model(latent_inputs, decoded, name="decoder")
    return sequence_autoencoder , encoder

def windwowIdentificationModel():
    timesteps = 1  # Length of your sequences
    input_dim = 5000
    latent_dim = 5000
    features = 7
    output_dim = 2000
    inputs = keras.Input(shape=(features, input_dim))


    encoded = keras.layers.Bidirectional(layers.LSTM(500, return_sequences=True, activation='tanh'))(inputs)


    encoded = TimeDistributed(layers.Dense(latent_dim, ))(encoded)
    # TimeDistributed(layers.Dense(latent_dim, ))(encoded)
    # layers.LSTM(latent_dim, return_sequences=True )(encoded)#

    latent_inputs = keras.Input(shape=(features, latent_dim,))

    # decoded = layers.LSTM(1000, return_sequences=True, name='dec2', activation='tanh')(latent_inputs)

    decoded = keras.layers.Bidirectional(layers.LSTM(500,  name='dec4', return_sequences=True, activation='tanh'))(latent_inputs)

    decoded = TimeDistributed(layers.Dense(latent_dim, ))(decoded)
    # decoded = layers.Dense(input_dim, activation="relu", )(decoded)

    #sequence_autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)
    decoder = keras.Model(latent_inputs, decoded, name="decoder")

    #print(encoder.summary())
    #print(decoder.summary())
    return  encoder , decoder

def baselineModel(newDimension=None):
    #latent_dim = 100 if newDimension==None else newDimension
    latent_dim = 7
    features = 7
    input_dim = 1000 if newDimension==None else newDimension

    encoder_inputs = keras.Input(shape=(n_steps,features ))

    encoded = layers.LSTM(500, activation='relu',return_sequences=True, name='lstmENC1')(encoder_inputs)
    #encoded = layers.LSTM(300, activation='relu', return_sequences=True, name='lstmENC2')(encoded)
    encoded = layers.LSTM(latent_dim, activation='relu',name='lstmENC')(encoded)
    encoded = RepeatVector(n_steps)(encoded)

    encoder = keras.Model(encoder_inputs, encoded , name="encoder")


    latent_inputs = keras.Input(shape=(n_steps,latent_dim))


    decoded = layers.LSTM(latent_dim, activation='relu',return_sequences=True, name='lstmDEC')(latent_inputs)
    #decoded = layers.LSTM(300, activation='relu', return_sequences=True, name='lstmDEC1')(decoded)
    decoded = layers.LSTM(500, activation='relu', return_sequences=True, name='lstmDEC2')(decoded)
    #decoded = layers.LSTM(500, activation='relu',  kernel_initializer='he_uniform',return_sequences=True, name='lstmDEC1')(decoded)


    decoded = TimeDistributed(layers.Dense(features, activation="relu", ))(decoded)

    decoder = keras.Model(latent_inputs, decoded, name="decoder")

    return encoder, decoder


def VAE_getMemoryWindowBetweenTaskA_TaskB():

    seqLSTMBtr = seqLSTMB.transpose()#.reshape(n_steps,9,1000)
    seqLSTMAtr = seqLSTMA.transpose()#.reshape(n_steps,9,1000)

    encoder , decoder  = VAE_windwowIdentificationModel()
    windAE = VAE(encoder, decoder)
    windAE.compile(optimizer=keras.optimizers.Adam())


    windAE.fit(seqLSTMAtr,seqLSTMBtr,epochs=200,)

    #windAE_.fit(seqLSTMAtr,seqLSTMAtr,epochs=100,)

    #windEncoder = windAE.layers[:-2]
    encodedTimeSeries = np.round(windAE.encoder.predict(seqLSTMAtr),2)
    encodedTimeSeriesReshaped = encodedTimeSeries.reshape(1000,9)

    selectiveMemory_ExtractedOfTaskA = seqLSTMA[[k for k in range(0,1000) if encodedTimeSeriesReshaped[:,k].all()>0]]

    return selectiveMemory_ExtractedOfTaskA

def getMemoryWindowBetweenTaskA_TaskB(lenSeq):

    seqLSTMAB = np.append(seqLSTMA,seqLSTMB, axis=0)
    seqABmean = []
    #for i in range(0,1000):
        #seqLSTMABmean = (seqLSTMA[i] + seqLSTMAB[i])/2
        #seqABmean.append(seqLSTMABmean)
    #seqLSTMAB = np.array(seqABmean)
    seqLSTMABtr = seqLSTMAB.transpose()#.reshape(7,n_steps,2000)
    min_max_scalerAB =  MinMaxScaler()
    X_train_normAB = min_max_scalerAB.fit_transform(seqLSTMABtr)
    #seqAB =  X_train_normAB.reshape(7, n_steps ,2000)

    seqLSTMBtr = seqLSTMB.transpose()#.reshape(n_steps,9,1000)
    seqLSTMAtr = seqLSTMA.transpose()#.reshape(n_steps,9,1000)

    #seqLSTMBtrReshaped = seqLSTMBtr.reshape(lenSeq, n_steps,7 )
    #seqLSTMAtrReshaped = seqLSTMAtr.reshape(lenSeq, n_steps,7 )

    min_max_scalerB = MinMaxScaler()
    min_max_scalerA  = MinMaxScaler()
    X_train_normB = min_max_scalerB.fit_transform(seqLSTMBtr)
    X_train_normA = min_max_scalerA.fit_transform(seqLSTMAtr)

    seqA = X_train_normA.reshape( n_steps,7 , lenSeq)
    seqB = X_train_normB.reshape( n_steps,7 , lenSeq)


    #seqAE , enc = lstmAUTO()
    #seqAE.compile(optimizer=keras.optimizers.Adam(),loss='mse')
    #seqAE.fit(seqA,seqB)

    encoderA , decoderA  = windwowIdentificationModel()
    #encoderA.compile(optimizer=keras.optimizers.Adam(),loss='mse')
    #encoderA.fit(seqA, seqA,epochs=30)
    #encoderA.fit(seqB, seqB,epochs=30)
    windAEa = LSTM_AE_IW(encoderA, decoderA)
    windAEa.compile(optimizer=keras.optimizers.Adam())

    es = keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, mode='min')
    #seqAB = np.append(seqA[0],seqB[0],axis=0)
    #seqAB = seqAB.reshape(1, 7 , lenSeq*2)
    windAEa.fit(seqA,seqB ,epochs=30,)

    '''encoderB, decoderB = windwowIdentificationModel()
    windAEb = LSTM_AE_IW(encoderB, decoderB)
    windAEb.compile(optimizer=keras.optimizers.Adam())

    windAEb.fit(seqB, seqA, epochs=30, )'''
    ##########################################

    '''encoderB , decoderB  = windwowIdentificationModel()
    #encoderA.compile(optimizer=keras.optimizers.Adam(),loss='mse')
    #encoderA.fit(seqA, seqA,epochs=30)
    windAEb = LSTM_AE_IW(encoderB, decoderB)
    windAEb.compile(optimizer=keras.optimizers.Adam())

    es = keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, mode='min')
    windAEb.fit(seqB,seqA,epochs=30, )'''

    encodedTimeSeriesA = np.round(windAEa.encoder.predict(seqA),2)

    #encodedTimeSeriesB = np.round(windAEb.encoder.predict(seqB), 2)

    encodedTimeSeriesReshapedA = encodedTimeSeriesA.reshape(lenSeq, 7)
    #encodedTimeSeriesReshapedB = encodedTimeSeriesB.reshape(7, 1000)
    arr = encodedTimeSeriesA[0] > 0
    indicesOfA = [k for k in range(0, lenSeq) if arr[:, k].all() == True]

    selectiveMemory_ExtractedOfTaskA =  seqLSTMA[indicesOfA]

    #arr = encodedTimeSeriesB[0] > 0
    #indicesOfB = [k for k in range(0, lenSeq) if arr[:, k].all() == True]
    #selectiveMemory_ExtractedOfTaskB = seqLSTMB[indicesOfB]

    #plt.plot(np.linspace(0, 5000, 5000), seqLSTMA[:, 3], color='red')
    #plt.plot(decodedA[:,3])
    #plt.plot(np.linspace(5000, 10000, 5000), seqLSTMB[:, 3], color='blue')
    #plt.plot(indicesOfA, selectiveMemory_ExtractedOfTaskA[:,3], color='green')
    #plt.show()

    with open('./AE_files/selectiveMemory_ExtractedOfTaskA.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(selectiveMemory_ExtractedOfTaskA)):
            data_writer.writerow(
                [selectiveMemory_ExtractedOfTaskA[i][0], selectiveMemory_ExtractedOfTaskA[i][1],
                 selectiveMemory_ExtractedOfTaskA[i][2],selectiveMemory_ExtractedOfTaskA[i][3],
                 selectiveMemory_ExtractedOfTaskA[i][4],
                 selectiveMemory_ExtractedOfTaskA[i][5],
                 selectiveMemory_ExtractedOfTaskA[i][6],
                 ])
    scaledArr_A = seqLSTMA.transpose() + (seqLSTMA.transpose() * encodedTimeSeriesA[0])
    scaledArr_A = np.array(scaledArr_A).transpose()

    scaledArr_B = seqLSTMB.transpose() + (seqLSTMB.transpose() * encodedTimeSeriesA[0])
    scaledArr_B = np.array(scaledArr_B).transpose()
    return scaledArr_A
        #np.append(scaledArr_A,scaledArr_B, axis=0)
    #np.append(scaledArr_A,scaledArr_B, axis=0)
        #np.append(selectiveMemory_ExtractedOfTaskA,selectiveMemory_ExtractedOfTaskB,axis=0)

def trainAE():
    encoder, decoder = baselineModel()
    lstm_autoencoderInit = LSTM_AE(encoder,decoder,  )
    lstm_autoencoderInit.compile(optimizer=keras.optimizers.Adam())


    for task in tasks:

        lstm_autoencoderInit.fit(task,task, epochs=100 )

    X_train_normB = min_max_scaler.fit_transform(seqLSTMB)
    x_test_encoded = lstm_autoencoderInit.encoder.predict(X_train_normB.reshape(1000,n_steps,9))
    decSeqInit = lstm_autoencoderInit.decoder.predict(x_test_encoded)
    decSeqInit =  decSeqInit.reshape(1000,7)
    decSeqInit  = min_max_scaler.inverse_transform(decSeqInit)
    scoreAE = np.linalg.norm(seqLSTMB.reshape(1000,7)-decSeqInit,axis=0)
        #
    print("AE Score :  " + str(scoreAE))

    with open('./AE_files/decodedSeaquenceofNewTaskWithoutMemory.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(decSeqInit)):
            data_writer.writerow(
                [decSeqInit[i][0], decSeqInit[i][1], decSeqInit[i][2], decSeqInit[i][3],
                 decSeqInit[i][4],
                 decSeqInit[i][5], decSeqInit[i][6], decSeqInit[i][7], decSeqInit[i][8]
                 ])
    return scoreAE

def trainAE_withMemoryOfPrevTask_inLS(selectiveMemoryExtractedOfTaskA):

    decSeqWithDiffWindow = []
    mem = selectiveMemoryExtractedOfTaskA.shape[1]
    #addMemoryToNewTask = np.append(selectiveMemoryExtractedOfTaskA, seqLSTMB, axis=0)

    #newDimension = len(addMemoryToNewTask)

    min_max_scaler = MinMaxScaler()
    #X_train_norm = min_max_scaler.fit_transform(addMemoryToNewTask)
    X_train_normB = min_max_scaler.fit_transform(seqLSTMB)

    X_train_normB = X_train_normB.reshape(len(X_train_normB),n_steps,9)

    #addMemoryToNewTask = X_train_norm.reshape(len(X_train_norm),n_steps,9)

    encoder, decoder = baselineModel(mem)
    lstm_autoencoderMem = LSTM_AE(encoder, decoder,)
    lstm_autoencoderMem.compile(optimizer=keras.optimizers.Adam())
    lstm_autoencoderMem.fit(X_train_normB, X_train_normB, epochs=10)


    X_train_normB = X_train_normB.reshape(1000, n_steps, 7)

    x_test_encoded = lstm_autoencoderMem.encoder.predict(X_train_normB)

    newEncodedLatentSpace = np.append(x_test_encoded,selectiveMemoryExtractedOfTaskA.reshape(mem, 1,9), axis=0)

    decSeqMem = lstm_autoencoderMem.decoder.predict(newEncodedLatentSpace)
    #decSeqMem = decSeqMem.reshape(1000,9)
    decSeqMem = decSeqMem.reshape(len(decSeqMem), 7)
    decSeqMem = min_max_scaler.inverse_transform(decSeqMem)

    decSeqWithDiffWindow.append(decSeqMem)
        # score , acc = lstm_autoencoder.evaluate(tasks[i], tasks[i], epochs=10)
    #diff = newDimension - 1000
    decSeq = decSeqMem.reshape(len(decSeqMem),9)#[diff:,:]


    with open('./AE_files/decodedSequenceofNewTaskWithMemory.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(decSeq)):
            data_writer.writerow(
                [decSeq[i][0], decSeq[i][1], decSeq[i][2], decSeq[i][3],
                 decSeq[i][4],
                 decSeq[i][5], decSeq[i][6], decSeq[i][7], decSeq[i][8]
                 ])

    scoreAE_mem = np.linalg.norm(seqLSTMB.reshape(1000,9) - decSeq[:-mem],axis=0)
        #scipy.stats.entropy(seqLSTMB.reshape(1000,9) ,qk=decSeq)
        #
    print("AE Score with mem window: " + str(len(selectiveMemoryExtractedOfTaskA)) + "  " + str(scoreAE_mem))
    return scoreAE_mem

def trainAE_withMemoryOfPrevTask(selectiveMemoryExtractedOfTaskA):
    decSeqWithDiffWindow = []
    addMemoryToNewTask = np.append(selectiveMemoryExtractedOfTaskA, seqLSTMB, axis=0)

    newDimension = len(addMemoryToNewTask)

    min_max_scaler = MinMaxScaler()
    X_train_norm = min_max_scaler.fit_transform(addMemoryToNewTask)
    X_train_normB = min_max_scaler.fit_transform(seqLSTMB)

    addMemoryToNewTask = X_train_norm.reshape(len(X_train_norm),n_steps,9)



    encoder, decoder = baselineModel(newDimension)
    lstm_autoencoderMem = LSTM_AE(encoder, decoder,)
    lstm_autoencoderMem.compile(optimizer=keras.optimizers.Adam())
    lstm_autoencoderMem.fit(addMemoryToNewTask, addMemoryToNewTask, epochs=100)

    x_test_encoded = lstm_autoencoderMem.encoder.predict(X_train_normB.reshape(1000,n_steps,9))
    decSeqMem = lstm_autoencoderMem.decoder.predict(x_test_encoded)
    decSeqMem = decSeqMem.reshape(1000,9)
    decSeqMem = min_max_scaler.inverse_transform(decSeqMem)

    decSeqWithDiffWindow.append(decSeqMem)
        # score , acc = lstm_autoencoder.evaluate(tasks[i], tasks[i], epochs=10)
    diff = newDimension - 1000
    decSeq = decSeqMem.reshape(1000,9)#[diff:,:]


    with open('./AE_files/decodedSequenceofNewTaskWithMemory.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(decSeq)):
            data_writer.writerow(
                [decSeq[i][0], decSeq[i][1], decSeq[i][2], decSeq[i][3],
                 decSeq[i][4],
                 decSeq[i][5], decSeq[i][6], decSeq[i][7], decSeq[i][8]
                 ])

    scoreAE_mem = np.linalg.norm(seqLSTMB.reshape(1000,9) - decSeq,axis=0)
        #scipy.stats.entropy(seqLSTMB.reshape(1000,9) ,qk=decSeq)
        #
    print("AE Score with mem window: " + str(diff) + "  " + str(scoreAE_mem))
    return scoreAE_mem
#####################
def plotDistributions(seqA,seqB ,var):

    dfA = pd.DataFrame({'stwA':seqA[:,3]})
    dfB = pd.DataFrame({'stwB': seqB[:, 3]})

    sns.displot(dfA, x='stwA')
    sns.displot(dfB, x='stwB')

    plt.show()

def trainAE_withRollingWIndowOfPrevTask(seqLSTMA, seqLSTMB):
    memWindow = [5, 10, 15, 20]

    memoryEncoderLayers = []
    mem = 0
    decSeqWithDiffWindow = []

    for mem in memWindow:

        encoder, decoder = baselineModel()
        lstm_autoencoderMem = LSTM_AE(encoder, decoder,)
        lstm_autoencoderMem.compile(optimizer=keras.optimizers.Adam())

        memoryOfPrevTask = seqLSTMA[-mem:]
        addMemoryToNewTask = np.append(memoryOfPrevTask, seqLSTMB, axis=0)

        lstm_autoencoderMem.fit(addMemoryToNewTask, addMemoryToNewTask, epochs=30)

        x_test_encoded = lstm_autoencoderMem.encoder.predict(seqLSTMB)
        decSeqMem = lstm_autoencoderMem.decoder.predict(x_test_encoded)

        decSeqWithDiffWindow.append(decSeqMem)
        # score , acc = lstm_autoencoder.evaluate(tasks[i], tasks[i], epochs=10)

        scoreAE_mem = np.linalg.norm(seqLSTMB - decSeqMem)
        print("AE Score with mem window: " + str(mem) + "  " + str(scoreAE_mem))

        #lstm_autoencoder.fit(tasks[i], tasks[i], epochs=10)

    seqLSTMB = seqLSTMB.reshape(seqLSTMB.shape[0],9)
    stwOr = seqLSTMB[:,3]

    stwDec = decSeqInit
    stwDec = stwDec.reshape(stwDec.shape[0],9)
    stwDec = stwDec[:,3]

    #dfStwOr = pd.DataFrame({"time":np.linspace(0,len(stwOr),len(stwOr)),"stwOr":stwOr})
    '''stwDec5 = decSeqWithDiffWindow[0]
    stwDec5 = stwDec5.reshape(1000,9)
    stwDec5 =stwDec5[:,3]
    
    stwDec10 = decSeqWithDiffWindow[1]
    stwDec10 = stwDec10.reshape(1000,9)
    stwDec10 =stwDec10[:,3]
    
    
    stwDec15 = decSeqWithDiffWindow[2]
    stwDec15 = stwDec15.reshape(1000,9)
    stwDec15 =stwDec15[:,3]
    
    
    stwDec20 = decSeqWithDiffWindow[3]
    stwDec20 = stwDec20.reshape(1000,9)
    stwDec20 = stwDec20[:,3]
    
    stwDec25 = decSeqWithDiffWindow[4]
    stwDec25 = stwDec25.reshape(1000,9)
    stwDec25 = stwDec25[:,3]'''

    stwDec30 = decSeqWithDiffWindow[0]
    stwDec30 = stwDec30.reshape(stwDec30.shape[0],9)
    stwDec30 = stwDec30[:,3]

    dfStwDec = pd.DataFrame({"time":np.linspace(0,len(stwDec),len(stwDec)),"stw Original":stwOr,
                            "stw Decoded without memory":stwDec,
                             "stw Decoded with 30min memory of previous task":stwDec30,
                            },)

    ''' "stw Decoded with memory 10min":stwDec10,
                             "stw Decoded with memory 15min":stwDec15,
                             "stw Decoded with memory 20min":stwDec20,
                             "stw Decoded with memory 25min":stwDec25,
                             "stw Decoded with memory 30min":stwDec30,'''
    dfStwDec.plot(kind='kde')
    plt.xlim(min(stwDec),max(stwDec))
    plt.show()

def trainingBaselinesForFOCestimation(seqLSTMA, seqLSTMB, memory, alg ):
    #X_train, X_test, y_train, y_test = train_test_split(seqLSTMB[:, :8], seqLSTMB[:, 8], test_size=0.2, random_state=42)

    lrA = LinearRegression()

    xa = seqLSTMA[:, :6]
    ya = seqLSTMA[:, 6]
    lrA.fit(xa, ya)


    print("Memory of taskA: "+str(len(memory)))
    lrBase = LinearRegression()

    xa = seqLSTMA[:, :6]
    ya = seqLSTMA[:, 6]
    lrBase.fit(xa, ya)


    xb = seqLSTMB[:,:6]
    yb = seqLSTMB[:, 6]

    lrBase.fit(xb, yb)
    score = lrBase.score(xb, yb)
    #print(str(score))
    maesa_ = []
    maesb_ = []
    for i in range(0, len(xb)):
        y_hat = lrBase.predict(xb[i].reshape(1,-1))[0]
        err = abs(y_hat - yb[i])
        maesb_.append(err)

    for i in range(0, len(xa)):
        y_hat = lrBase.predict(xa[i].reshape(1, -1))[0]
        err = abs(y_hat - ya[i])
        maesa_.append(err)

    #print(metrics.r2_score(yb, maes_))
    #plt.plot(np.linspace(0, len(yb), len(yb)), yb)
    #plt.plot(np.linspace(0, len(maes_), len(maes_)), maes_)
    #plt.show()
    print("Score for A without memory:" + str(np.mean(maesa_)))
    print("Score for B without memory:" + str(np.mean(maesb_))+"\n")
    lr = LinearRegression()

    #stackedTrain = np.append(X_train,y_train.reshape(-1,1),axis=1)


    seqBwithMem = np.append(memory, seqLSTMB.reshape(-1,7),axis=0)
        #np.append(memory, seqLSTMB.reshape(-1,7),axis=0)

    xb_mem = seqBwithMem[:, :6]
    yb_mem = seqBwithMem[:, 6]

    lrA = LinearRegression()
    lrA.fit(xb_mem, yb_mem)

    maesa_s = []
    maesb_s = []
    for i in range(0,len(xb)):
        y_hat = lrA.predict(xb[i].reshape(1,-1))[0]
        err = abs(y_hat- yb[i])
        maesb_s.append(err)

    for i in range(0,len(xa)):
        y_hat = lrA.predict(xa[i].reshape(1,-1))[0]
        err = abs(y_hat - ya[i])
        maesa_s.append(err)

    print("Score for A with selective memory:" + str(np.mean(maesa_s)))
    print("Score for B with selective memory:" + str(np.mean(maesb_s)))
    print("Score difference: " +str(abs(np.mean(maesb_s) -np.mean(maesa_s)) )+"\n")

    lr = LinearRegression()
    lrA = LinearRegression()

    xa = seqLSTMA[:, :6]
    ya = seqLSTMA[:, 6]
    lrA.fit(xa, ya)

    maesa_f = []
    maesb_f = []
    stacked =  np.append( seqLSTMA.reshape(-1,7),seqLSTMB,axis=0)
    xStacked = stacked[:,:6]
    yStacked = stacked[:, 6]
    lrA.fit(xStacked, yStacked)
    for i in range(0,len(xb)):
        y_hat = lrA.predict(xb[i].reshape(1,-1))[0]
        err = abs(y_hat- yb[i])
        maesb_f.append(err)

    for i in range(0,len(xa)):
        y_hat = lrA.predict(xa[i].reshape(1,-1))[0]
        err = abs(y_hat - ya[i])
        maesa_f.append(err)

    print("Score for A with  full memory:" + str(np.mean(maesa_f)))
    print("Score for B with  full memory:" + str(np.mean(maesb_f)) + "\n")

    df = pd.DataFrame.from_dict({"ScoreA without memory": np.mean(maesa_) ,
                       "ScoreB without memory": np.mean(maesb_),
                       "ScoreA with selective memory": np.mean(maesa_s),
                       "ScoreB with selective memory": np.mean(maesb_s),
                       "ScoreA with full memory": np.mean(maesa_f),
                       "ScoreB with full memory": np.mean(maesb_f),
                       },orient='index')
    #df.to_csv('./AE_files/'+alg+'.csv')
    return df
#######################################################
def runAlgorithmsforEvaluation( alg, seqLen):
    memories = None
    if alg!='RND':
        dfs = []
        memories = []
        for k in range(0, 1):
            memory = getMemoryWindowBetweenTaskA_TaskB(seqLen)
            memories.append(len(memory))
            df = trainingBaselinesForFOCestimation(seqLSTMA, seqLSTMB, memory, alg)
            dfs.append(df)

        merged = pd.concat(dfs)
        merged.to_csv('./AE_files/' + alg + '.csv')

    if alg =='RND':
        dfs = []
        alg = 'RND'
        for k in range(0, 5):
            randomMemoryofTaskA = seqLSTMA[np.random.randint(seqLSTMA.shape[0], size=len(memories[k])), :]

            df = trainingBaselinesForFOCestimation(seqLSTMA, seqLSTMB, randomMemoryofTaskA, alg)
            dfs.append(df)

        merged = pd.concat(dfs)
        merged.to_csv('./AE_files/' + alg + '.csv')

    return memories

def main():

    #plotDistributions(seqLSTMA,seqLSTMB,'foc')
    #return
    # memory = pd.read_csv('./AE_files/selectiveMemory_ExtractedOfTaskA.csv', ).values
    lenMemories = runAlgorithmsforEvaluation('LR', len(seqLSTMA))
    pd.DataFrame({'memories':lenMemories}).to_csv('./AE_files/lenMemories.csv')
    lr = pd.read_csv('./AE_files/LR.csv').values

    fullMemMeanErrA = np.mean(np.array([k for k in lr if k[0] == 'ScoreA with full memory'])[:, 1])
    fullMemMeanErrB = np.mean(np.array([k for k in lr if k[0] == 'ScoreB with full memory'])[:, 1])

    selMemMeanErrA = np.mean(np.array([k for k in lr if k[0] == 'ScoreA with selective memory'])[:, 1])
    selMemMeanErrB = np.mean(np.array([k for k in lr if k[0] == 'ScoreB with selective memory'])[:, 1])

    withoutMemMeanErrA = np.mean(np.array([k for k in lr if k[0] == 'ScoreA without memory'])[:, 1])
    withoutMemMeanErrB = np.mean(np.array([k for k in lr if k[0] == 'ScoreB without memory'])[:, 1])

    print("full MEM error for A and B "+str((fullMemMeanErrA + fullMemMeanErrB)/2))
    print("sel MEM error for A and B " + str((selMemMeanErrA + selMemMeanErrB) / 2))
    print("without MEM error for A and B " + str((withoutMemMeanErrA + withoutMemMeanErrB) / 2))


    #print(str(len(memory)))
    #memory = pd.read_csv('./AE_files/selectiveMemory_ExtractedOfTaskA.csv',).values
    #print(str(len(memory)))
    return
    scoresAE=[]
    scoresAE_mem = []
    '''for k in range(0,5):
       scoreAE =  trainAE()
       scoresAE.append(np.round(scoreAE,2))

    scores = pd.DataFrame({'scoreAE':scoresAE,})
    scores.to_csv('./AE_files/scoresAE.csv')'''

    #return

    '''for k in range(0,5):
        scoreAE_mem = trainAE_withMemoryOfPrevTask(memory)
        scoresAE_mem.append(np.round(scoreAE_mem,2))

    scores = pd.DataFrame({'scoreAE_mem':scoresAE_mem})
    scores.to_csv('./AE_files/scoresAE_mem.csv')'''

    stwMem = memory[:,3]
    stwA = seqLSTMA[:,3]
    while len(stwMem)<=len(stwA):
        stwMem = np.append(stwMem,stwMem)



    foc12 = [(itm, 'stw of task A') for itm in stwA]
    foc23 = [(itm, 'stw of memory extracted task A') for itm in stwMem]

    joinedFoc = foc12 + foc23

    df = pd.DataFrame(data=joinedFoc,
                      columns=['stw', 'Original / Extracted memory'])
    # df.Zip = df.Zip.astype(str).str.zfill(5)

    # plt.title('FOC distributions')

    sns.displot(df, x="stw", hue='Original / Extracted memory', kind="kde", multiple="stack")

    #stwMem = np.append(stwMem, np.nan(len(stwA)-len(stwMem)))

    dfStwDec = pd.DataFrame({"time":np.linspace(0,len(stwA),len(stwA)),
                                "stw of task A":stwA,

                                },)

    dfStwMem = pd.DataFrame({"time": np.linspace(min(stwMem), len(stwMem), len(stwMem)),
                             "stw of memory extracted task A": stwMem,

                             }, )

    #dfStwDec.plot(kind='kde')
    sns.displot(dfStwDec, x="stw of task A", kind="kde")
    sns.displot(dfStwMem, x="stw of memory extracted task A", kind="kde")
    #plt.xlim(min(stwA),max(stwA))

    #dfStwMem.plot(kind='kde')
    plt.xlim(min(stwMem),max(stwMem))
    plt.legend()
    plt.show()

# # ENTRY POINT
if __name__ == "__main__":
    main()