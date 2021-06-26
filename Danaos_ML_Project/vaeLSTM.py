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
from AttentionDecoder import AttentionDecoder

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def LSTMVAEKL():
    latent_dim = 3

    model = Sequential()

    encoder_inputs = keras.Input(shape=(5, 9))
    x = layers.LSTM(latent_dim, activation='relu',name='lstmENC')(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    z = RepeatVector(5)(z)
    #x = RepeatVector(5)(x)
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")



    #model.add(LSTM(200, activation='relu', input_shape=(5,9),name='lstmENC'))
    #model.add(LSTM(10, activation='relu',name='lstmENC1'))

    #model.add(RepeatVector(5))


    latent_inputs = keras.Input(shape=(5,latent_dim,))
    #x = RepeatVector(latent_inputs)
    x = layers.LSTM(latent_dim, activation='relu', return_sequences=True,name='lstmDEC')(latent_inputs)

    decoder_outputs = TimeDistributed(layers.Dense(9,  activation="relu", ))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    #model.add(LSTM(200, activation='relu', return_sequences=True,name='lstmDEC'))
    #model.add(TimeDistributed(Dense(9,name='denseDEC')))
    #model.compile(optimizer='adam', loss='mse')


def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
    #+tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)

n_steps = 5




def baselineModel():
    latent_dim = 100

    memory = []

    model = Sequential()

    encoder_inputs = keras.Input(shape=(n_steps, 9))
    encoded = layers.LSTM(latent_dim, activation='relu',return_sequences=True, name='lstmENC')(encoder_inputs)
    encoded = layers.LSTM(latent_dim-20, activation='relu', name='lstmENC1')(encoded)

    encoded = RepeatVector(n_steps)(encoded)

    encoder = keras.Model(encoder_inputs, encoded , name="encoder")


    latent_inputs = keras.Input(shape=(n_steps,latent_dim-20))




    decoded = layers.LSTM(latent_dim, activation='relu', return_sequences=True, name='lstmDEC')(latent_inputs)
    decoded = layers.LSTM(latent_dim-20, activation='relu', return_sequences=True, name='lstmDEC1')(decoded)


    decoded = TimeDistributed(layers.Dense(9, activation="relu", ))(decoded)

    decoder = keras.Model(latent_inputs, decoded, name="decoder")

    return encoder, decoder




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
            data = data[0]

            encoderOutput = self.encoder(data)
            reconstruction = self.decoder(encoderOutput)

            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction) + keras.losses.kullback_leibler_divergence(data,reconstruction)

            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        #self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),

        }

##train

#vae = VAE(encoder, decoder)


data = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/mappedDataNew.csv').values
trData = np.array(np.append(data[:,8].reshape(-1,1), np.asmatrix([data[:,10], data[:,11], data[:,12], data[:,22],
                                           data[:,1] ,data[:,26],data[:,27],data[:,15]]).T,axis=1)).astype(float)#data[:,26],data[:,27]

trData = np.nan_to_num(trData)
trData = np.array([k for k in trData if  str(k[0])!='nan' and  float(k[2])>0 and float(k[4])>0 and (float(k[3])>=8 ) and float(k[8])>0  ]).astype(float)

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



raw_seq = trData[:4000]
seqLSTM = raw_seq
# split into samples
#seqLSTM = split_sequence(raw_seq, n_steps)
#seqLSTM = seqLSTM.reshape(-1,n_steps,9)

dataLength = len(seqLSTM)
seqLSTMA = seqLSTM[:2000]

seqLSTMAmem = split_sequence(seqLSTMA,n_steps)
seqLSTMAmem = seqLSTMAmem.reshape(-1,n_steps,9)

seqLSTMB = seqLSTM[2000:4000]

tasks = [seqLSTMA, seqLSTMB]

encoder, decoder = baselineModel()
lstm_autoencoderInit = LSTM_AE(encoder,decoder,  )
lstm_autoencoderInit.compile(optimizer=keras.optimizers.Adam())
#lstm_autoencoder.fit(seqLSTM,seqLSTM, epochs=100, )
#print(scipy.stats.ks_2samp(seqLSTMA, seqLSTMB))
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,restore_best_weights=True)
memWindow = [5,10,15,20]

memoryEncoderLayers=[]
mem =0


lstm_autoencoderInit.fit(seqLSTMA,seqLSTMB)


for task in tasks:
    lstm_autoencoderInit.fit(task,task, epochs=30)


x_test_encoded = lstm_autoencoderInit.encoder.predict(seqLSTMB)
decSeqInit = lstm_autoencoderInit.decoder.predict(x_test_encoded)
scoreAE = np.linalg.norm(seqLSTMB - decSeqInit)
print("AE Score :  " + str(scoreAE))
decSeqWithDiffWindow = []
for mem in memWindow:

    #encoder, decoder = baselineModel()
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
#for mem in memWindow:
    #memoryTask = seqLSTMA[-mem:]
    #addMemory = np.append(memoryTask,seqLSTMB,axis=0)
    #lstm_autoencoder.fit(addMemory, seqLSTMB[0:len(addMemory)],epochs=30,)
#for task in tasks:
    #model.fit(task, task,epochs=30,callbacks=[es])
    #print("END TASK----------------------\n\n")