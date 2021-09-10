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
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
import random
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
import matplotlib.patches as patches
import pyearth as sp
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

def ApEn(U, m, r) -> float:
    """Approximate_entropy."""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

trData = getData()

raw_seq = trData#[:4000]
seqLSTM = raw_seq
# split into samples
#seqLSTM = split_sequence(raw_seq, n_steps)
#seqLSTM = seqLSTM.reshape(-1,n_steps,9)
n_steps = 20
dataLength = len(seqLSTM)
tasksA=[]
tasksB=[]
tasksAMem=[]
tasksBMem=[]
lenS = 1000
startA = 13000
startB = 50000
for i in range(0,1):

    seqLSTMA = seqLSTM[startA:lenS+startA]
    seqLSTMAmem = split_sequence(seqLSTMA, n_steps)
    seqLSTMAmem = seqLSTMAmem.reshape(-1, n_steps, 7)

    seqLSTMB = seqLSTM[startB:lenS + startB]
    seqLSTMBmem = split_sequence(seqLSTMB, n_steps)
    seqLSTMBmem = seqLSTMBmem.reshape(-1, n_steps, 7)
    #tasksA.append(seqLSTM[startA:lenS+startA])
    #tasksB.append(seqLSTM[startB:lenS+startB])
    tasksAMem.append(seqLSTMAmem)
    tasksBMem.append(seqLSTMBmem)

    tasksA.append(seqLSTMA)
    tasksB.append(seqLSTMB)
    startA += lenS
    startB += lenS

seqLSTMA = seqLSTM[21000:22000]

seqLSTMAmem = split_sequence(seqLSTMA,n_steps)
seqLSTMA = seqLSTMAmem.reshape(-1,n_steps,7)

seqLSTMB = seqLSTM[60000:61000]

seqLSTMBmem = split_sequence(seqLSTMB,n_steps)
seqLSTMB = seqLSTMBmem.reshape(-1,n_steps,7)

#scipy.stats.ks_2samp(seqLSTMA, seqLSTMB)
# call MinMaxScaler object
min_max_scaler = MinMaxScaler()
seqAScaled = []
for i in range(0,len(seqLSTMA)):
    seqScaled =  min_max_scaler.fit_transform(seqLSTMA[i])
    seqAScaled.append(seqScaled)

X_train_normA = np.array(seqAScaled)
#X_train_normA = min_max_scaler.fit_transform(seqLSTMA)
#X_train_normB = min_max_scaler.fit_transform(seqLSTMB)
min_max_scaler = MinMaxScaler()
seqBScaled = []
for i in range(0,len(seqLSTMB)):
    seqScaled =  min_max_scaler.fit_transform(seqLSTMB[i])
    seqBScaled.append(seqScaled)

X_train_normB = np.array(seqBScaled)
#apEn = ApEn(tasksB[0][:,3],2,3)
#print(str(apEn))

tasks = [X_train_normA.reshape(len(seqLSTMA),n_steps,7), X_train_normB.reshape(len(seqLSTMB),n_steps,7)]
def pointInRect(point,rect):
    x1, y1 = rect.xy
    w, h = rect.get_bbox().width , rect.get_bbox().height
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

def boxCountingMethod(sequence,ylabel,width = None, height = None):
    width = 100
    height = 1
    ax = plt.gca()
    colors =['blue','red']
    colorSeq = ['green', 'black']
    labels = ['TASKA','TASKB']
    counters = []
    rectsDicts = []
    for i in  range(0,len(sequence)):

        rectsDict = {'rects':[]}
        for k in range(0,lenS,width): ##columns
            for n in range(0,int(max(sequence[i]))+1): #rows
                rect = patches.Rectangle((k, n), width, height, linewidth=1, edgecolor=colors[i], facecolor='none')
                item = {}
                item['visited'] = False
                item['rect'] = rect
                rectsDict['rects'].append(item)

                #ax.add_patch(rect)
        plt.plot(sequence[i],c=colorSeq[i],label=labels[i])
        # Add the patch to the Axes
        rectsVisited = []
        #plt.show()
        counter = 0
        for n in range(0,len(sequence[i])):
            for rect in rectsDict['rects']:
                if pointInRect((n,sequence[i][n]),rect['rect']) and rect['visited']==False:
                    counter += 1
                    rect['visited'] = True
                    rectsDict['rects'].remove(rect)
                    rectsVisited.append(rect['rect'])
                    break
        counters.append(counter)
        for rect in rectsVisited:
            ax.add_patch(rect)

        plt.xlabel('time')
        plt.ylabel(ylabel)
        rectsDicts.append(rectsVisited)
    plt.title('Fractal dimension with box counting method for task A: '+ str(counters[0]) +" for task B: "+str(counters[1])+ " with τ = "+str(width)+" and α = " + str(height))
    plt.legend()


    dictA  = rectsDicts[0]
    dictB = rectsDicts[1]
    filteredSeqAs = []
    indices = []


    k=0
    xis = []
    for i in range(0,len(dictB)): ##iterate rects of B
            filteredSeqA = []
            indices = []
            xi = int(dictB[i].xy[0])
            xi_hat = int(dictB[i].get_bbox().width)
            xis.append(xi)
            if xi != xis[i-1] or i ==0:
                for n in range(xi,xi+xi_hat):
                    if pointInRect((n, sequence[0][n]), dictB[i]) == False :
                         #ax.add_patch(rect['rect'])
                         flag = True
                         filteredSeqA.append(sequence[0][n])

                         indices.append(n)
            #ax.add_patch(plt.plot(filteredSeqA))
            filteredSeqAs.append(indices)

    print(len(indices))

    #plt.show()
    return tasksA[0][list(np.concatenate(filteredSeqAs).astype(int))]
    #print("Fractal dimension: "+str(counter))


#memoryA = boxCountingMethod([tasksA[0][:,3],tasksB[0][:,3]],'stw')
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

            reconstruction_loss = keras.losses.mean_squared_error(data,reconstruction) +\
            keras.losses.kullback_leibler_divergence(encoderOutput, reconstruction)
            #keras.losses.mean_squared_error(data,reconstruction) +\




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
                                  +  keras.losses.kullback_leibler_divergence(encoderOutput,reconstruction)
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
    timesteps = n_steps # Length of your sequences
    input_dim = lenS - n_steps
    latent_dim = lenS - n_steps
    features = 7
    output_dim = lenS
    #inputs = keras.Input(shape=(features, input_dim))
    inputs = keras.Input(shape=(features, timesteps))#(7,20)

    # keras.layers.Bidirectional(
    encoded = (layers.LSTM(n_steps - int(n_steps/2), return_sequences=True, activation='tanh'))(inputs)

    encoded =  layers.LSTM(timesteps, return_sequences=True, activation='tanh')(encoded)
    #encoded = TimeDistributed(layers.Dense(latent_dim, ))(encoded)
    # TimeDistributed(layers.Dense(latent_dim, ))(encoded)
    # layers.LSTM(latent_dim, return_sequences=True )(encoded)#
    latent_inputs = keras.Input(shape=( features, timesteps))
    # decoded = layers.LSTM(1000, return_sequences=True, name='dec2', activation='tanh')(latent_inputs)

    decoded =  (layers.LSTM(n_steps - int(n_steps/2),  name='dec4', return_sequences=True, activation='tanh'))(latent_inputs)

    decoded =  layers.LSTM(timesteps, return_sequences=True, activation='tanh')(decoded)
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

def getMemoryWindowBetweenTaskA_TaskB(lenSeq,seqLSTMA, seqLSTMB):

    '''seqLSTMAB = np.append(seqLSTMA,seqLSTMB, axis=0)

    seqLSTMABtr = seqLSTMAB.transpose()#.reshape(7,n_steps,2000)
    min_max_scalerAB =  MinMaxScaler()

    seqLSTMBtr = seqLSTMB.transpose()#.reshape(n_steps,9,1000)
    seqLSTMAtr = seqLSTMA.transpose()#.reshape(n_steps,9,1000)


    min_max_scalerB = MinMaxScaler()
    min_max_scalerA  = MinMaxScaler()
    X_train_normB = min_max_scalerB.fit_transform(seqLSTMBtr)
    X_train_normA = min_max_scalerA.fit_transform(seqLSTMAtr)

    seqA = X_train_normA.reshape( n_steps,7 , lenSeq)
    seqB = X_train_normB.reshape( n_steps,7 , lenSeq)'''


    ######################################################################
    min_max_scalerA = MinMaxScaler()
    seqAScaled = []
    for i in range(0, len(seqLSTMA)):
        seqScaled = min_max_scalerA.fit_transform(seqLSTMA[i])
        #seqScaled = seqLSTMA[i]
        seqScaled = seqScaled.transpose()
        seqAScaled.append(seqScaled)

    X_train_normA = np.array(seqAScaled)
    # X_train_normA = min_max_scaler.fit_transform(seqLSTMA)
    # X_train_normB = min_max_scaler.fit_transform(seqLSTMB)
    min_max_scalerB = MinMaxScaler()
    seqBScaled = []
    for i in range(0, len(seqLSTMB)):
        seqScaled = min_max_scalerB.fit_transform(seqLSTMB[i])
        #seqScaled = seqLSTMB[i]
        seqScaled = seqScaled.transpose()
        seqBScaled.append(seqScaled)

    X_train_normB = np.array(seqBScaled)
    #seqA, seqB =  X_train_normA.reshape( n_steps,7 , lenSeq-n_steps), X_train_normB.reshape( n_steps,7 , lenSeq-n_steps)
    seqA, seqB = X_train_normA, X_train_normB
    #seqAE , enc = lstmAUTO()
    #seqAE.compile(optimizer=keras.optimizers.Adam(),loss='mse')
    #seqAE.fit(seqA,seqB)

    encoderA , decoderA  = windwowIdentificationModel()
    windAEa = LSTM_AE_IW(encoderA, decoderA)
    windAEa.compile(optimizer=keras.optimizers.Adam())

    es = keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, mode='min')

    windAEa.fit(seqA, seqB , epochs=30,)#batch_size=20

    '''encoderB, decoderB = windwowIdentificationModel()
    windAEb = LSTM_AE_IW(encoderB, decoderB)
    windAEb.compile(optimizer=keras.optimizers.Adam())

    windAEb.fit(seqB, seqA, epochs=30, )'''

    '''encoderB, decoderB = windwowIdentificationModel()
    windAEb = LSTM_AE_IW(encoderB, decoderB)
    windAEb.compile(optimizer=keras.optimizers.Adam())

    windAEb.fit(seqB, seqA, epochs=50, )'''
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



    arr = encodedTimeSeriesA[0] > 0
    #indicesOfA = [k for k in range(0, lenSeq) if arr[:, k].all() == True]

    #selectiveMemory_ExtractedOfTaskA =  seqLSTMA[indicesOfA]
    batches = []
    for i in range(0,len(encodedTimeSeriesA)):
        #batch = min_max_scalerA.inverse_transform((seqA[i] + (seqA[i] * encodedTimeSeriesA[i])).transpose())
        #batches.append(batch)

        seq_a = min_max_scalerA.inverse_transform(seqA[i].transpose())
        batch = (seq_a + (seq_a * encodedTimeSeriesA[i].transpose()))
        batches.append(batch)

    selectiveMemory_ExtractedOfTaskA = np.array(batches)

    '''batches = []
    for i in range(0, len(encodedTimeSeriesB)):

        seq_b = min_max_scalerB.inverse_transform(seqB[i].transpose())
        batch = (seq_b + (seq_b * encodedTimeSeriesB[i].transpose()))
        batches.append(batch)'''

    selectiveMemory_ExtractedOfTaskB = np.array(batches)

    #scaledArr_A = seqLSTMA.transpose() + (seqLSTMA.transpose() * encodedTimeSeriesA[0])


    #plt.plot(np.linspace(0, 5000, 5000), seqLSTMA[:, 3], color='red')
    #plt.plot(decodedA[:,3])
    #plt.plot(np.linspace(5000, 10000, 5000), seqLSTMB[:, 3], color='blue')
    #plt.plot(indicesOfA, selectiveMemory_ExtractedOfTaskA[:,3], color='green')
    #plt.show()

    '''with open('./AE_files/selectiveMemory_ExtractedOfTaskA.csv', mode='w') as data:
        data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(selectiveMemory_ExtractedOfTaskA)):
            data_writer.writerow(
                [selectiveMemory_ExtractedOfTaskA[i][0], selectiveMemory_ExtractedOfTaskA[i][1],
                 selectiveMemory_ExtractedOfTaskA[i][2],selectiveMemory_ExtractedOfTaskA[i][3],
                 selectiveMemory_ExtractedOfTaskA[i][4],
                 selectiveMemory_ExtractedOfTaskA[i][5],
                 selectiveMemory_ExtractedOfTaskA[i][6],
                 ])'''
    #scaledArr_A = seqLSTMA.transpose() + (seqLSTMA.transpose() * encodedTimeSeriesA[0])
    #scaledArr_A = np.array(scaledArr_A).transpose()

    #scaledArr_B = seqLSTMB.transpose() + (seqLSTMB.transpose() * encodedTimeSeriesB[0])
    rndIndices = []
    rndLen = random.randint(300,900)
    for k in range(0,rndLen):
        rndIndices.append(random.randint(0, lenSeq-1))
    #selectiveMemory_ExtractedOfTaskA = seqLSTMA[rndIndices]
    #scaledArr_B = np.array(scaledArr_B).transpose()
    return selectiveMemory_ExtractedOfTaskA, None
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

def plotErrBetweenTasks(seqLSTMA, seqLSTMB, lrSpeedSel , lrSpeedFull):

    '''visualizeLen = 1000
    plt.plot(np.linspace(0, visualizeLen, visualizeLen), y_hatsa[:visualizeLen], color='red',label='predicted')
    plt.plot(np.linspace(0, visualizeLen, visualizeLen),ya[:visualizeLen],color='green',label='TASKA')
    plt.plot(np.linspace(visualizeLen, visualizeLen*2, visualizeLen), y_hatsb[:visualizeLen], color='red')
    plt.plot(np.linspace(visualizeLen, visualizeLen*2, visualizeLen),yb[:visualizeLen],'blue',label='TASKB')'''
    plot='B'
    if plot =='A':
        i = np.min(seqLSTMA[:,3])
        maxSpeedA = np.max(seqLSTMA[:,3])
        focsPLot = []
        speedsPlot = []
        ranges = []
        featuresPlot = []
        while i <= maxSpeedA:
            # workbook._sheets[sheet].insert_rows(k+27)
            focArray = np.array([k for k in seqLSTMA if float(k[3]) >= i - 0.25 and float(k[3]) <= i + 0.25])
            # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
            '''meanFoc = np.mean(focArray[:, 8])
            stdFoc = np.std(focArray[:, 8])
            speedFoc = np.array([k for k in focArray if k[8] >= (meanFoc - (3 * stdFoc)) and k[8] <= (meanFoc + (3 * stdFoc))])'''

            if focArray.__len__() > 0:
                focsPLot.append(focArray.__len__())
                featuresPlot.append(np.mean(focArray[:,:6],axis=0))
                speedsPlot.append(np.mean(focArray[:,3],axis=0))
                ranges.append(np.mean(focArray[:, 6]))
                # lrSpeedFoc.fit(focArray[:,5].reshape(-1, 1), focArray[:,8].reshape(-1, 1))
            i += 0.5
            # k += 1

        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(focsPLot)

        # p2 = np.poly1d(np.polyfit(xi, yi, 2,w=focsPLot),)
        #xiBlue = np.linspace(9, 20, 20 - 9)

        # plt.plot([], [], '.', xp, p2(xp))
        speedList = [8, 9, 10, 11, 12, 13, 14]

        plt.plot(xi, lrSpeedSel.predict(np.concatenate(featuresPlot).reshape(-1, 6)), c='blue')
        plt.plot(xi, yi, c='red')

        plt.scatter(xi, yi, s=zi / 10, c="red", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi)), np.ceil(max(xi)) + 1, 1))
        # plt.yticks(np.arange(min(yi), max(yi) + 1, 5))
        plt.xlabel("Speed (knots)")
        plt.ylabel("FOC (MT / day)")
        plt.title("Density plot for taskA", loc="center")
        plt.show()

    i = np.min(seqLSTMB[:, 3])
    maxSpeedB = np.max(seqLSTMB[:, 3])
    focsPLot = []
    speedsPlot = []
    ranges = []
    featuresPlot = []
    while i <= maxSpeedB:
        # workbook._sheets[sheet].insert_rows(k+27)
        focArray = np.array([k for k in seqLSTMB if float(k[3]) >= i - 0.25 and float(k[3]) <= i + 0.25])
        # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
        '''meanFoc = np.mean(focArray[:, 8])
        stdFoc = np.std(focArray[:, 8])
        speedFoc = np.array([k for k in focArray if k[8] >= (meanFoc - (3 * stdFoc)) and k[8] <= (meanFoc + (3 * stdFoc))])'''

        if focArray.__len__() > 0:
            focsPLot.append(focArray.__len__())
            featuresPlot.append(np.mean(focArray[:, :6], axis=0))
            speedsPlot.append(np.mean(focArray[:, 3], axis=0))
            ranges.append(np.mean(focArray[:, 6]))
            # lrSpeedFoc.fit(focArray[:,5].reshape(-1, 1), focArray[:,8].reshape(-1, 1))
        i += 0.5
        # k += 1

    xi = np.array(speedsPlot)
    yi = np.array(ranges)
    zi = np.array(focsPLot)

    plt.plot(xi, lrSpeedSel.predict(np.concatenate(featuresPlot).reshape(-1, 6)), c='blue',label='Base learner with selected mem')
    plt.plot(xi, lrSpeedFull.predict(np.concatenate(featuresPlot).reshape(-1, 6)), c='green',label='Base learner with full mem')
    plt.plot(xi, yi, c='red')

    plt.scatter(xi, yi, s=zi / 10, c="red", alpha=0.4, linewidth=4)
    # plt.xticks(np.arange(np.floor(min(xi)), np.ceil(max(xi)) + 1, 1))
    # plt.yticks(np.arange(min(yi), max(yi) + 1, 5))
    plt.xlabel("Speed (knots)")
    plt.ylabel("FOC (MT / day)")
    plt.legend()
    plt.title("Density plot for task B", loc="center")
    plt.show()


def baselineLearner():
    # create model
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(50,input_shape=(n_steps, 6,), ))  # return_sequences=True

    model.add(keras.layers.Dense(20, ))

    model.add(keras.layers.Dense(10, ))

    model.add(keras.layers.Dense(1))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam())  # experimental_run_tf_function=False )
    # print(model.summary())

    return model


def trainingBaselinesForFOCestimation(seqLSTMA, seqLSTMB, memoryA, memoryB, alg ):
    #X_train, X_test, y_train, y_test = train_test_split(seqLSTMB[:, :8], seqLSTMB[:, 8], test_size=0.2, random_state=42)
    #SplineRegression = sp.Earth()
    lrA = LinearRegression()

    xa = seqLSTMA[:, :6]
    ya = seqLSTMA[:, 6]
    lrA.fit(xa, ya)


    print("Memory of taskA: "+str(len(memoryA)))
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
    scoreAWM = lrBase.score(xa, ya)
    scoreBWM = lrBase.score(xb, yb)

    print("Score for A without memory:" + str(np.mean(maesa_)))
    print("Score for B without memory:" + str(np.mean(maesb_)))
    print("R2 Score for A without memory:" + str(scoreAWM))
    print("R2 Score for B without memory:" + str(scoreBWM) + "\n")

    lrAB = baselineLearner()
    batchesX = []
    batchesY = []
    for i in range(0,len(memoryA)):

        batchesX.append(memoryA[i][:,:6])
        batchesY.append(memoryA[i][n_steps-1,6])

    batchesX = np.array(batchesX)
    batchesY = np.array(batchesY)

    batchesXa = []
    batchesYa = []
    for i in range(0, len(seqLSTMAmem)):
        batchesXa.append(seqLSTMAmem[i][:, :6])
        batchesYa.append(seqLSTMAmem[i][n_steps - 1, 6])

    batchesXa = np.array(batchesXa)
    batchesYa = np.array(batchesYa)

    batchesXb = []
    batchesYb = []
    for i in range(0, len(seqLSTMBmem)):
        batchesXb.append(seqLSTMBmem[i][:, :6])
        batchesYb.append(seqLSTMBmem[i][n_steps - 1, 6])

    batchesXb = np.array(batchesXb)
    batchesYb = np.array(batchesYb)
    lstmX = np.append(batchesX, batchesXa,axis=0)
    lstmX = np.append(lstmX, batchesXb, axis=0)
    lstmY = np.append(batchesY, batchesYa)
    lstmY = np.append(lstmY, batchesYb)

    lrAB.fit(lstmX, lstmY, epochs=20)
    #lrAB.layers[0].set_weights([weights, np.array([0] * (genModelKnots - 1))])
    recosntructA = []
    for i in range(0,len(memoryA)):

         if i ==0:
            recon = memoryA[i][0]
         else:
             recon = memoryA[i][0]
         recosntructA.append(recon)
    #seqAscaled = np.append(np.array(recosntructA[0]).reshape(n_steps, 7), np.array(recosntructA[1:]).reshape(-1, 7), axis=0)
    seqAscaled = np.array(recosntructA)

    #if memoryB!=None:
    '''recosntructB = []
    for i in range(0, len(memoryB)):

        if i == 0:
            recon = memoryB[i][0]
        else:
            recon = memoryB[i][0]
        recosntructB.append(recon)'''

    #seqBscaled = np.append(np.array(recosntructB[0]).reshape(n_steps, 7), np.array(recosntructB[1:]).reshape(-1, 7),
                               #axis=0)
    #seqBscaled = np.array(recosntructB)

    seqBwithMem = np.append(seqAscaled, seqLSTMB.reshape(-1,7),axis=0)

    #seqAwithMem = np.append(seqBscaled, seqLSTMA.reshape(-1,7),axis=0)

    seqABwithMem = np.append(seqAscaled, seqLSTMB.reshape(-1, 7), axis=0)



    xab_mem = seqABwithMem[:, :6]
    yab_mem = seqABwithMem[:, 6]

    #lrAB.fit(xab_mem, yab_mem)

    lrB = LinearRegression()
    xb_mem = seqBwithMem[:, :6]
    yb_mem = seqBwithMem[:, 6]

    #lrB = LinearRegression()


    lrA = LinearRegression()
    #xa_mem = seqAwithMem[:, :6]
    #ya_mem = seqAwithMem[:, 6]

    # lrB = LinearRegression()
    ###fit ensembles
    lrA.fit(xb_mem, yb_mem)
    #lrA.fit(xa_mem, ya_mem)

    ###############################
    #lrB.fit(xa_mem, ya_mem)
    lrB.fit(xb_mem, yb_mem)

    maesa_s = []
    maesb_s = []
    y_hatsa = []
    y_hatsb = []
    for i in range(0,len(batchesXb)):
        y_hatb = lrAB.predict(batchesXb[i].reshape(1,n_steps,6))[0][0]
        y_hatsb.append(y_hatb)
        err = abs(y_hatb - batchesYb[i])
        maesb_s.append(err)

    for i in range(0,len(batchesXa)):
        y_hata = lrAB.predict(batchesXa[i].reshape(1,n_steps,6))[0][0]
        y_hatsa.append(y_hata)
        err = abs(y_hata - batchesYa[i])
        maesa_s.append(err)

    scoreASel = lrA.score(xa, ya)
    scoreBSel = lrB.score(xb, yb)
    print("Score for A with selective memory:" + str(np.mean(maesa_s)))
    print("Score for B with selective memory:" + str(np.mean(maesb_s)))
    print("R2 Score for A with selective memory:" + str(scoreASel))
    print("R2 Score for B with selective memory:" + str(scoreBSel))
    print("Score difference: " +str(abs(np.mean(maesb_s) - np.mean(maesa_s)))+"\n")



    lrAfull = LinearRegression()
    xa = seqLSTMA[:, :6]
    ya = seqLSTMA[:, 6]
    #lrAfull.fit(xa, ya)

    maesa_f = []
    maesb_f = []
    stacked =  np.append( seqLSTMA.reshape(-1,7),seqLSTMB,axis=0)
    #stacked = seqLSTMA
    xStacked = stacked[:,:6]
    yStacked = stacked[:, 6]
    lrAfull.fit(xStacked, yStacked)
    #################################################



    for i in range(0,len(xb)):
        y_hat = lrAfull.predict(xb[i].reshape(1,-1))[0]
        err = abs(y_hat- yb[i])
        maesb_f.append(err)

    for i in range(0,len(xa)):
        y_hat = lrAfull.predict(xa[i].reshape(1,-1))[0]
        err = abs(y_hat - ya[i])
        maesa_f.append(err)

    scoreAFull = lrAfull.score(xa, ya)
    scoreBFull = lrAfull.score(xb, yb)

    print("Score for A with  full memory:" + str(np.mean(maesa_f)))
    print("Score for B with  full memory:" + str(np.mean(maesb_f)) )
    print("R2 Score for A with full memory:" + str(scoreAFull) )
    print("R2 Score for B wuth full memory:" + str(scoreBFull) + "\n")

    #plotErrBetweenTasks(seqLSTMA, seqLSTMB, lrA, lrAfull)

    df = pd.DataFrame.from_dict({"ScoreA without memory": np.mean(maesa_) ,
                       "ScoreB without memory": np.mean(maesb_),
                        "R2A without memory:": scoreAWM,
                        "R2B without memory:": scoreBWM,
                       "ScoreA with selective memory": np.mean(maesa_s),
                       "ScoreB with selective memory": np.mean(maesb_s),
                        "R2A with selective memory:": scoreASel,
                        "R2B with selective memory:": scoreBSel,
                       "ScoreA with full memory": np.mean(maesa_f),
                       "ScoreB with full memory": np.mean(maesb_f),
                        "R2A with full memory:": scoreAFull,
                        "R2B with full memory:": scoreBFull,
                       },orient='index')
    #df.to_csv('./AE_files/'+alg+'.csv')
    return df
#######################################################
def runAlgorithmsforEvaluation( alg, seqLen):
    memories = None
    if alg!='RND':
        dfs = []
        memories = []
        for k in range(0, len(tasksA)):
            memoryA, memoryB = getMemoryWindowBetweenTaskA_TaskB(seqLen,tasksAMem[k], tasksBMem[k])
            #memories.append(len(memory))
            df = trainingBaselinesForFOCestimation(tasksA[k], tasksB[k], memoryA, memoryB, alg)
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
    lenMemories = runAlgorithmsforEvaluation('LR', lenS )
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