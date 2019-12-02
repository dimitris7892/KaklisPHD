import sklearn.linear_model as sk
from sklearn.linear_model import  LinearRegression
import pyearth as sp
import numpy as np
import numpy.linalg
import sklearn.ensemble as skl
from scipy.spatial import Delaunay
import random
#from sklearn.cross_validation import train_test_split
import parameters
import itertools
#from sklearn.model_selection import KFold as kf
from scipy import spatial
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import RandomizedSearchCV
from basis_expansions1 import NaturalCubicSpline
from sklearn.preprocessing import StandardScaler
from gekko import GEKKO
import  matplotlib.pyplot as plt
from scipy.interpolate import BivariateSpline
import tensorflow as tf
from tensorflow import keras
#from sklearn.model_selection import KFold
#import pydot
#import graphviz
import scipy.stats as st
#from tensorflow.python.tools import inspect_checkpoint as chkp
from time import time
import metrics
from sklearn.cluster import KMeans

tf.executing_eagerly()

class BasePartitionModeler:
    def createModelsFor(self,partitionsX, partitionsY, partition_labels):
        pass

    def getBestModelForPoint(self, point):
        #mBest = None
        mBest=None
        dBestFit = 0
        # For each model
        for m in self._models:
            # If it is a better for the point
            dCurFit = self.getFitnessOfModelForPoint(m, point)
            if dCurFit > dBestFit:
                # Update the selected best model and corresponding fit
                dBestFit = dCurFit
                mBest = m

        if mBest==None:
            return self._models[0]
        else: return mBest

    def getFitForEachPartitionForPoint(self, point, partitions):
        # For each model
        fits=[]
        for m in range(0, len(partitions)):
            # If it is a better for the point
            dCurFit = self.getFitnessOfPoint(partitions, m, point)
            fits.append(dCurFit)


        return fits

    def getBestPartitionForPoint(self, point,partitions):
            # mBest = None
            mBest = None
            dBestFit = 0
            # For each model
            for m in range(0,len(partitions)):
                # If it is a better for the point
                dCurFit = self.getFitnessOfPoint(partitions,m, point)
                if dCurFit > dBestFit:
                    # Update the selected best model and corresponding fit
                    dBestFit = dCurFit
                    mBest = m

            if mBest == None:
                return 0,0
            else:
                return mBest , dBestFit

    def  getFitnessOfModelForPoint(self, model, point):
        return 0.0

    def getFitnessOfPoint(self,partitions ,cluster, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[ cluster ]) - point))


class TriInterpolantModeler(BasePartitionModeler):

    def getFitnessOfPoint(self,partitions ,cluster, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[ cluster ]) - point))


class LinearRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):
        # Init result model list
        models = []
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = sk.LinearRegression()
            # Fit to data
            curModel.fit(partitionsX[idx], partitionsY[idx])
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[curModel] = partitionsX[idx]

        # Update private models
        self._models = models

        # Return list of models
        return models , numpy.empty ,numpy.empty , None

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

class TensorFlowWD(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels, tri, X, Y):

        X = np.array(np.concatenate(partitionsX))
        Y = np.array(np.concatenate(partitionsY))

        models = []
        self.ClustersNum = len(partitionsX)
        self.modelId = -1
        partition_labels = partitionsX
        # Init model to partition map
        self._partitionsPerModel = {}

        def SplinesCoef(partitionsX, partitionsY):

            model = sp.Earth(use_fast=True)
            model.fit(partitionsX, partitionsY)

            return model.coef_

        def getFitnessOfPoint(partitions, cluster, point):
            return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster]) - point))

        def baseline_modelDeepCl():
            # create model
            model = keras.models.Sequential()

            model.add(keras.layers.Dense(len(partition_labels) * 3, input_shape=(3,)))
            model.add(keras.layers.Dense(len(partition_labels) * 2, input_shape=(3,)))
            model.add(keras.layers.Dense(len(partition_labels), input_shape=(3,)))
            # model.add(keras.layers.Dense(10, input_shape=(2,)))
            # model.add(keras.layers.Dense(5, input_shape=(2,)))
            #model.add(keras.layers.Activation(custom_activation))
            model.add(keras.layers.Dense(1, ))  # activation=custom_activation
            # model.add(keras.layers.Activation(custom_activation))
            # model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss='kld', optimizer=keras.optimizers.Adam())
            return model

        def baseline_model():
            # create model
            model = keras.models.Sequential()

            # model.add(keras.layers.Dense(len(partition_labels)*2, input_shape=(2,)))
            # model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,) ))
            # model.add(keras.layers.Activation(custom_activation2))
            model.add(keras.layers.Dense(len(partition_labels), input_shape=(6,)))
            #model.add(keras.layers.Dense(10, input_shape=(6,)))
            # model.add(MyLayer(5))
            # model.add(keras.layers.Dense(15, input_shape=(2,)))

            # model.add(keras.layers.Dense(genModelKnots, input_shape=(2,)))
            model.add(keras.layers.Dense(5,))
            # model.add(keras.layers.Activation(custom_activation2(inputs=model.layers[2].output, modelId=1)))
            #model.add(keras.layers.Activation(custom_activation2))
            model.add(keras.layers.Dense(1, ))  # activation=custom_activation
            # model.add(keras.layers.Activation(custom_activation))
            # model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            return model

        seed = 7
        numpy.random.seed(seed)

        #partitionsX = X
        #partitionsY = Y
        # weights = preTrainedWeights()

        dims = [2, 1000, 500, 200, 100, len(partition_labels)]
        init = keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in',
                                                  distribution='uniform')
        # pretrain_optimizer =keras.optimizers.SGD(lr=1, momentum=0.9)
        seed = 7
        numpy.random.seed(seed)

        #partitionsX.reshape(-1, 2)

        estimator = baseline_model()
        estimator.fit(X, Y, epochs=200, validation_split=0.33)

        def insert_intermediate_layer_in_keras(model, layer_id, new_layer):

            layers = [l for l in model.layers]

            x = layers[0].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)

                x = layers[i](x)
            # x = new_layer(x)
            new_model = keras.Model(inputs=model.input, outputs=x)
            # new_model.add(new_layer)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            # .add(x)
            return new_model

        def replace_intermediate_layer_in_keras(model, layer_id, new_layer):

            layers = [l for l in model.layers]

            x = layers[0].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)
                else:
                    x = layers[i](x)

            new_model = keras.Model(inputs=model.input, outputs=x)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            return new_model

        NNmodels = []
        import csv
        for idx, pCurLbl in enumerate(partition_labels):
            #
            self.modelId = idx + 1
            modelId = idx

            estimatorCl=replace_intermediate_layer_in_keras(estimator, -1 ,keras.layers.Dense(5))
            estimatorCl.fit(np.array(partitionsX[idx]), np.array(partitionsY[idx]), epochs=30)
            estimatorCl.save("estimatorCl_"+str(idx)+".h5")
            NNmodels.append(estimatorCl)
            with open('./cluster_' + str(idx) + '_.csv', mode='w') as data:
                for k in range(0,len(partitionsX[idx])):
                    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow([partitionsX[idx][k][0],partitionsX[idx][k][1],partitionsX[idx][k][2],partitionsX[idx][k][3],partitionsX[idx][k][4],partitionsX[idx][k][5] ])
        # Update private models
        # models=[]
        # models.append(estimator)
        #NNmodels.append(estimator)
        self._models = NNmodels

        # Return list of models
        return estimator, None, numpy.empty, numpy.empty, estimator, partitionsX

class TensorFlowW(BasePartitionModeler):


    def initNN(self,X_data,input_dim):
        #X=np.concatenate(partitionsX)
        W_1 = tf.Variable(tf.random_uniform([ input_dim, 10 ]))
        #W_1 = np.array(W_1).reshape(-1,2)
        #weightsShape = len(W_1[0])
        #W_1=tf.Variable(np.float32(W_1[0]))
        b_1 = tf.Variable(tf.zeros([ 10 ]))
        layer_1 = tf.add(tf.matmul(X_data,W_1),b_1)
        #layer_1 = tf.add( W_1, b_1)
        layer_1 = tf.nn.relu(layer_1)
        # layer 1 multiplying and adding bias then activation function

        W_2 = tf.Variable(tf.random_uniform([ 10,10 ]))
        b_2 = tf.Variable(tf.zeros([10 ]))
        layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
        layer_2 = tf.nn.relu(layer_2)
        # layer 2 multiplying and adding bias then activation function

        W_3 = tf.Variable(tf.random_uniform([10,10 ]))
        b_3 = tf.Variable(tf.zeros([ 10 ]))
        layer_3 = tf.add(tf.matmul(layer_2, W_3), b_3)
        layer_3 = tf.nn.relu(layer_3)
        #layer 3 multiplying and adding bias then activation function
        W_O = tf.Variable(tf.random_uniform([10, 1 ]))
        b_O = tf.Variable(tf.zeros([ 1 ]))
        output = tf.add(tf.matmul(layer_1, W_O), b_O)


        return  output
        # O/p layer multiplying and adding bias then activation function
        # notice output layer has one node only since performing #regression
        #return output

    def initNN_2(self,x,weights,biases):
        # Hidden layer with RELU activation
        h_layer_1 = tf.add(tf.matmul(x, weights[ 'h1' ]), biases[ 'h1' ])
        out_layer_1 = tf.sigmoid(h_layer_1)
        # Output layer with linear activation
        h_out = tf.matmul(out_layer_1, weights[ 'out' ]) + biases[ 'out' ]

        return h_out

    def createModelsForAUTO_ENC(self, partitionsX, partitionsY, partition_labels):
        x = partitionsX
        y = partitionsY

        #tri = Delaunay(x.reshape(-1,2))
        # x = x.reshape((x.shape[ 0 ], -1))
        n_clusters = 15
        # kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
        # y_pred_kmeans = kmeans.fit_predict(x)
        x = x[ 0, : ]
        x = x.reshape((x.shape[ 0 ], -1))
        ###########
        dims = [ x.shape[-1], 1000, 500, 200 , 100 ,n_clusters]
        init = keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in',
                                                  distribution='uniform')
        # pretrain_optimizer =keras.optimizers.SGD(lr=1, momentum=0.9)
        seed = 7
        numpy.random.seed(seed)

        # x=x.reshape(1,-1)
        # y=y.reshape(1,-1)
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)

        pretrain_epochs = 2
        batch_size = 100
        save_dir = '/home/dimitris/Desktop/results'
        autoencoder, encoder =self.autoencoder(dims, init=init)

        #keras.utils.plot_model(autoencoder, to_file='/home/dimitris/Desktop/autoencoder.png', show_shapes=True)
        #from IPython.display import Image
        #Image(filename='/home/dimitris/Desktop/autoencoder.png')
        # x=x[:,0]
        # X_train=X_train.reshape(-1,1)
        # y_train = y_train.reshape(-1,1)
        # X_test = X_test.reshape(-1, 1)
        # y_test = y_test.reshape(-1, 1)
        # x = x.reshape(1, -1)
        # y = y.reshape(1, -1)
        autoencoder.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
        #autoencoder.save_weights(save_dir + '/ae_weights.h5')

        #autoencoder.load_weights(save_dir + '/ae_weights.h5')

        return encoder.get_weights()[encoder.weights.__len__() - 1]

    def autoencoder(self,dims, act='relu', init='glorot_uniform'):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = len(dims) - 1
        #dims[ 0 ]=2
        # input
        input_img = keras.layers.Input(shape=(dims[ 0 ],), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(n_stacks - 1):
            x = keras.layers.Dense(dims[ i + 1 ], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
            # r=keras.layers.Reshape((dims[i+1],1,))(x)
            # lstm_x = keras.layers.LSTM(10, return_sequences=True, return_state=True)(r)
            # lstm_x_r =keras.layers.Reshape((dims[i+1],))(lstm_x)
        # hidden layer
        # encoded = keras.layers.Dense(dims[ -1 ], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(lstm_x[2])
        encoded = keras.layers.Dense(dims[ -1 ], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
        # hidden layer, features are extracted from here

        x = encoded

        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            x = keras.layers.Dense(dims[ i ], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = keras.layers.Dense(dims[ 0 ], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        return keras.models.Model(inputs=input_img, outputs=decoded, name='AE'), keras.models.Model(inputs=input_img,
                                                                                                    outputs=encoded,
                                                                                                    name='encoder')

    def createModelsForAUTO(self, partitionsX, partitionsY, partition_labels):

        models = [ ]
        # Init model to partition map
        #partitionsX = np.concatenate(partitionsX)
        #partitionsY = np.concatenate(partitionsY)

        self._partitionsPerModel = {}
        def SplinesCoef(partitionsX, partitionsY):

           model= sp.Earth(use_fast=True)
           model.fit(partitionsX,partitionsY)

           return model.coef_

        class MyLayer(tf.keras.layers.Layer):

            def __init__(self, output_dim):
                self.output_dim = output_dim
                super(MyLayer, self).__init__()

            def build(self, input_shape):

                # Create a trainable weight variable for this layer.
                self.kernel = self.add_weight(name='kernel',
                                              shape=(input_shape[ 1 ], self.output_dim),
                                              trainable=True)
                super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

            def call(self, x,mask=None):
                return keras.backend.dot(x, self.kernel)

            def compute_output_shape(self, input_shape):
                return (input_shape[ 0 ], self.output_dim)

        class DCEC(object):
            def __init__(self,
                         input_shape=(partitionsX.shape[0],partitionsX.shape[1], 1),
                         filters=[ 32, 64, 128, 10 ],
                         n_clusters=10,
                         alpha=1.0):

                super(DCEC, self).__init__()

                self.n_clusters = n_clusters
                self.input_shape = input_shape
                self.alpha = alpha
                self.pretrained = False
                self.y_pred = [ ]

                self.cae = CAE(input_shape, filters)
                hidden = self.cae.get_layer(name='embedding').output
                self.encoder =keras.models.Model(inputs=self.cae.input, outputs=hidden)

                # Define DCEC model
                clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)

                self.model = keras.models.Model(inputs=self.cae.input,
                                   outputs=[ clustering_layer, self.cae.output ])

            def pretrain(self, x, batch_size=256, epochs=10, optimizer='adam', save_dir='results/temp'):
                print('...Pretraining...')
                self.cae.compile(optimizer=optimizer, loss='mse')

                csv_logger =keras.callbacks.CSVLogger('/home/dimitris/Desktop' + '/pretrain_log.csv')

                # begin training
                t0 = time()
                # callbacks=[ csv_logger ]
                x=x[0:400]

                #self.cae.fit(x, x, batch_size=batch_size, epochs=epochs)
                print('Pretraining time: ', time() - t0)
                self.cae.save(save_dir + '/pretrain_cae_model.h5')
                print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
                self.pretrained = True

            def load_weights(self, weights_path):
                self.model.load_weights(weights_path)

            def extract_feature(self, x):  # extract features from before clustering layer
                return self.encoder.predict(x)

            def predict(self, x):
                q, _ = self.model.predict(x, verbose=0)
                return q.argmax(1)

            @staticmethod
            def target_distribution(q):
                weight = q ** 2 / q.sum(0)
                return (weight.T / weight.sum(1)).T

            def compile(self, loss=[ 'kld', 'mse' ], loss_weights=[ 1, 1 ], optimizer='adam'):
                self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

            def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
                    update_interval=140, cae_weights=None, save_dir='./results/temp'):

                print('Update interval', update_interval)
                save_interval = x.shape[ 0 ] / batch_size * 5
                print('Save interval', save_interval)

                # Step 1: pretrain if necessary
                t0 = time()
                if not self.pretrained and cae_weights is None:
                    print('...pretraining CAE using default hyper-parameters:')
                    print('   optimizer=\'adam\';   epochs=200')
                    self.pretrain(x, batch_size, save_dir=save_dir)
                    self.pretrained = True
                elif cae_weights is not None:
                    self.cae.load_weights(cae_weights)
                    print('cae_weights is loaded successfully.')

                # Step 2: initialize cluster centers using k-means
                t1 = time()
                print('Initializing cluster centers with k-means.')
                kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
                self.y_pred = kmeans.fit_predict(self.encoder.predict(x[ 0:400 ].reshape(1, 400, 1, 1))[ 0 ].reshape(-1,1))
                #self.y_pred = kmeans.fit_predict(self.encoder.predict(x[0,400].reshape(1,400,1,1)))
                y_pred_last = np.copy(self.y_pred)
                self.model.get_layer(name='clustering').set_weights([ kmeans.cluster_centers_ ])

                # Step 3: deep clustering
                # logging file
                import csv, os
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                logfile = open(save_dir + '/dcec_log.csv', 'w')
                logwriter = csv.DictWriter(logfile, fieldnames=[ 'iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr' ])
                logwriter.writeheader()

                t2 = time()
                loss = [ 0, 0, 0 ]
                index = 0
                for ite in range(int(maxiter)):
                    if ite % update_interval == 0:
                        q, _ = self.model.predict(x[0:400].reshape(1,400,1,1), verbose=0)
                        p = self.target_distribution(q)  # update the auxiliary target distribution p

                        # evaluate the clustering performance
                        self.y_pred = q.argmax(1)
                        if y is not None:
                            acc = np.round(metrics.acc(y, self.y_pred), 5)
                            nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                            ari = np.round(metrics.ari(y, self.y_pred), 5)
                            loss = np.round(loss, 5)
                            logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[ 0 ], Lc=loss[ 1 ], Lr=loss[ 2 ])
                            logwriter.writerow(logdict)
                            print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                        # check stop criterion
                        delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[ 0 ]
                        y_pred_last = np.copy(self.y_pred)
                        if ite > 0 and delta_label < tol:
                            print('delta_label ', delta_label, '< tol ', tol)
                            print('Reached tolerance threshold. Stopping training.')
                            logfile.close()
                            break

                    # train on batch
                    if (index + 1) * batch_size > x.shape[ 0 ]:
                        loss = self.model.train_on_batch(x=x[ index * batch_size:: ],
                                                         y=[ p[ index * batch_size:: ], x[ index * batch_size:: ] ])
                        index = 0
                    else:
                        loss = self.model.train_on_batch(x=x[ index * batch_size:(index + 1) * batch_size ],
                                                         y=[ p[ index * batch_size:(index + 1) * batch_size ],
                                                             x[ index * batch_size:(index + 1) * batch_size ] ])
                        index += 1

                    # save intermediate model
                    if ite % save_interval == 0:
                        # save DCEC model checkpoints
                        print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                        #self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

                    ite += 1

                # save the trained model
                logfile.close()
                print('saving model to:', save_dir + '/dcec_model_final.h5')
                #self.model.save_weights(save_dir + '/dcec_model_final.h5')
                t3 = time()
                print('Pretrain time:  ', t1 - t0)
                print('Clustering time:', t3 - t1)
                print('Total time:     ', t3 - t0)

        class ClusteringLayer(keras.layers.Layer):
            """
            Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
            sample belonging to each cluster. The probability is calculated with student's t-distribution.
            # Example
            ```
                model.add(ClusteringLayer(n_clusters=10))
            ```
            # Arguments
                n_clusters: number of clusters.
                weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
                alpha: parameter in Student's t-distribution. Default to 1.0.
            # Input shape
                2D tensor with shape: `(n_samples, n_features)`.
            # Output shape
                2D tensor with shape: `(n_samples, n_clusters)`.
            """

            def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
                if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                    kwargs[ 'input_shape' ] = (kwargs.pop('input_dim'),)
                super(ClusteringLayer, self).__init__(**kwargs)
                self.n_clusters = n_clusters
                self.alpha = alpha
                self.initial_weights = weights
                self.input_spec = keras.layers.InputSpec(ndim=2)

            def build(self, input_shape):
                assert len(input_shape) == 2
                input_dim = input_shape[ 1 ]
                self.input_spec =keras.layers.InputSpec(dtype=keras.backend.floatx(), shape=(None, input_dim))
                #self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),name='clusters')
                self.clusters = self.add_weight(shape=(self.n_clusters, 1), name='clusters')
                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights
                self.built = True

            def call(self, inputs, **kwargs):
                """ student t-distribution, as same as used in t-SNE algorithm.
                         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
                Arguments:
                    inputs: the variable containing data, shape=(n_samples, n_features)
                Return:
                    q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
                """
                q = 1.0 / (1.0 + (keras.backend.sum(keras.backend.square(keras.backend.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
                q **= (self.alpha + 1.0) / 2.0
                q = keras.backend.transpose(keras.backend.transpose(q) / keras.backend.sum(q, axis=1))
                return q

            def compute_output_shape(self, input_shape):
                assert input_shape and len(input_shape) == 2
                return input_shape[ 0 ], self.n_clusters

            def get_config(self):
                config = {'n_clusters': self.n_clusters}
                base_config = super(ClusteringLayer, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

        def stackedModels(members):
            for i,pCurLbl in  enumerate(partition_labels):
                model = members[ i ]
                for layer in model.model.layers:
                    # make not trainable
                    layer.trainable = False
                    # rename to avoid 'unique layer name' issue
                    #layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
                # define multi-headed input
            ensemble_visible = [ model.model.input for model in members ]
            # concatenate merge output from each model
            ensemble_outputs = [ model.model.output for model in members ]
            merge = keras.layers.concatenate(ensemble_outputs)
            hidden = keras.layers.Dense(10, activation='relu')(merge)
            output = keras.layers.Dense(1, activation='relu')(hidden)
            model = keras.models.Model(inputs=ensemble_visible, outputs=output)
            # plot graph of ensemble
            #keras.utils.plot_model(model, show_shapes=True, to_file='/home/dimitris/model_graph.png')
            #model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adagrad())
            return model


        def baseline_model1():
            model =keras.models.Sequential()

            # 1st convolution layer
            model.add(keras.layers.Conv2D(16, (3, 3)  # 16 is number of filters and (3, 3) is the size of the filter.
                             , padding='same', input_shape=(partitionsX.shape[0],1,1)))
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

            # 2nd convolution layer
            model.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

            # here compressed version

            # 3rd convolution layer
            model.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.UpSampling2D((2, 2)))

            # 4th convolution layer
            model.add(keras.layers.Conv2D(16, (3, 3), padding='same'))
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.UpSampling2D((2, 2)))

            model.add(keras.layers.Conv2D(1, (3, 3), padding='same'))
            model.add(keras.layers.Activation('sigmoid'))

            model.compile(optimizer='adadelta', loss='binary_crossentropy')

            return model


        def CAE(input_shape=(partitionsX[0:400].shape[0],partitionsX[0:400].shape[1], 1), filters=[ 32, 64, 128, 10 ]):
            model =keras.models.Sequential()
            if input_shape[ 0 ] % 8 == 0:
                pad3 = 'same'
            else:
                pad3 = 'valid'
            model.add(keras.layers.Conv2D(filters[ 0 ], 5, strides=2, padding='same', activation='relu', name='conv1',
                             input_shape=input_shape))

            model.add(keras.layers.Conv2D(filters[ 1 ], 5, strides=2, padding='same', activation='relu', name='conv2'))

            model.add(keras.layers.Conv2D(filters[ 2 ], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(units=filters[ 3 ], name='embedding'))
            model.add(
                keras.layers.Dense(units=filters[ 2 ] * int(input_shape[ 0 ] / 8) * int(input_shape[ 0 ] / 8), activation='relu'))

            model.add(keras.layers.Reshape((int(input_shape[ 0 ] / 8), 1, filters[ 2 ])))
            model.add(keras.layers.Conv2DTranspose(filters[ 1 ], 3, strides=1, padding=pad3, activation='relu', name='deconv3'))

            model.add(keras.layers.Conv2DTranspose(filters[ 0 ], 5, strides=1, padding='same', activation='relu', name='deconv2'))

            model.add(keras.layers.Conv2DTranspose(1, 1, strides=1, padding='same', name='deconv1'))
            #model.add(keras.layers.Reshape((int(input_shape[ 0 ] ), 1, 1)))
            model.summary()
            return model

        def baseline_model():
            # create model
           #SplineWeights = SplinesCoef(partitionsX[idx],partitionsY[idx])
            #SplineWeights=np.resize(SplineWeights, (3, 10))
            #SplineWeights=np.array([np.resize(SplineWeights[0],10),np.resize(SplineWeights[1],10),np.resize(SplineWeights[2],10)])
            # this is the size of our encoded representations
            encoding_dim = partitionsX.shape[0]/10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

            # this is our input placeholder
            input_img = keras.Input(shape=(partitionsX.shape[0],))
            # "encoded" is the encoded representation of the input
            encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_img)
            # "decoded" is the lossy reconstruction of the input
            decoded = keras.layers.Dense(partitionsX.shape[0], activation='sigmoid')(encoded)

            # this model maps an input to its reconstruction
            autoencoder = keras.models.Model(input_img, decoded)

            # this model maps an input to its encoded representation
            encoder = keras.models.Model(input_img, encoded)

            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = keras.Input(shape=(encoding_dim,))
            # retrieve the last layer of the autoencoder model
            decoder_layer = autoencoder.layers[ -1 ]
            # create the decoder model
            decoder = keras.models.Model(encoded_input, decoder_layer(encoded_input))

            # Compile model
            autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

            return autoencoder , encoder , decoder
        #################################################   clustering NN
        x=partitionsX
        y=partitionsY
        #x = np.append(x, np.asmatrix([ y ]).T, axis=1)
        #x = x.reshape((x.shape[ 0 ], -1))
        n_clusters=self.ClustersNum
        #kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
        #y_pred_kmeans = kmeans.fit_predict(x)
        #x = x[ 0, : ]
        #x = x.reshape((x.shape[ 0 ], -1))
        x = x.reshape((-1, x.shape[ 0 ]))
        ###########
        dims = [ x.shape[-1], 1000, 500, 200 , 100 ,n_clusters ]
        init =keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')
        #pretrain_optimizer =keras.optimizers.SGD(lr=1, momentum=0.9)
        seed = 7
        numpy.random.seed(seed)

        #x=x.reshape(1,-1)
        #y=y.reshape(1,-1)
        #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)

        pretrain_epochs = 3
        batch_size = 100
        pretrain_optimizer =keras.optimizers.SGD(lr=1, momentum=0.9)

        save_dir = '/home/dimitris/Desktop/results'
        autoencoder, encoder =self.autoencoder(dims, init=init)


        #keras.utils.plot_model(autoencoder, to_file='/home/dimitris/Desktop/autoencoder.png', show_shapes=True)
        #from IPython.display import Image
        #Image(filename='/home/dimitris/Desktop/autoencoder.png')
        x=x.reshape(-1,2)
        autoencoder.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
        #autoencoder.save_weights(save_dir + '/ae_weights.h5')

        #autoencoder.load_weights(save_dir + '/ae_weights.h5')

        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)

        #modelLayer =  keras.layers.Input(shape=(1,),name='inputModel')
        model = keras.models.Model(inputs=encoder.input, outputs=clustering_layer)

        model.compile(optimizer=keras.optimizers.SGD(0.01, 0.9),loss='kld')
                      #
        kmeans = KMeans(n_clusters=n_clusters,n_init=20)
        ##y_pred_kMeans = kmeans.fit_predict(encoder.predict(x)[0].reshape(-1,1))
        #y_pred = kmeans.fit_predict(x.reshape(-1, 1))
        y_pred = kmeans.fit_predict(encoder.predict(x)[0].reshape(-1,1))


        y_pred_last = np.copy(y_pred)
        ########SET K MEANS INITIAL WEIGHTS TO CLUSTERING LAYER
        model.get_layer(name='clustering').set_weights([ kmeans.cluster_centers_ ])

        loss = 0
        index = 0
        maxiter = 8000
        update_interval = 140
        index_array = np.arange(x.shape[ 0 ])
        tol = 0.001  # tolerance threshold to stop training
        y=None
        #x=x.reshape(1,-1)
        def target_distribution(q):
            weight = q ** 2 / q.sum(0)
            return (weight.T / weight.sum(1)).T
        #####start training
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = model.predict(x, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[ 0 ]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[ index * batch_size: min((index + 1) * batch_size, x.shape[ 0 ]) ]
            loss = model.train_on_batch(x=x[ idx ], y=p[ idx ])
            index = index + 1 if (index + 1) * batch_size <= x.shape[ 0 ] else 0
        ###########


        #model.save_weights(save_dir + '/DEC_model_final.h5')
        #model.load_weights(save_dir + '/DEC_model_final.h5')

        # Eval.
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p
        models.append(autoencoder)

        y_predDeepCl = q.argmax(1)

        DeepCLpartitionsX = [ ]
        DeepCLpartitionsY = [ ]
        DeepClpartitionLabels = [ ]
        # For each label
        x2 = partitionsX.reshape(-1, 2)
        y2 = partitionsY.reshape(-1, 1)
        for curLbl in np.unique(y_predDeepCl):
            # Create a partition for X using records with corresponding label equal to the current
            DeepCLpartitionsX.append(np.asarray(x2[ y_predDeepCl == curLbl ]))
            # Create a partition for Y using records with corresponding label equal to the current
            DeepCLpartitionsY.append(np.asarray(y2[ y_predDeepCl == curLbl ]))
            # Keep partition label to ascertain same order of results
            DeepClpartitionLabels.append(curLbl)

        self._models = models

        # Return list of models
        #return models, numpy.empty, numpy.empty, None
        return DeepCLpartitionsX, DeepCLpartitionsY, DeepClpartitionLabels
        # evaluate the clustering performance

        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)
        #########################################################################################
        ###########################################CONV2D##################
        # prepare the DCEC model
        dcec = DCEC(input_shape=(partitionsX[0:400].shape[0],partitionsX[0:400].shape[1], 1), filters=[ 32, 64, 128, 10 ], n_clusters=10)
        #plot_model(dcec.model, to_file=args.save_dir + '/dcec_model.png', show_shapes=True)
        dcec.model.summary()

        # begin clustering.
        optimizer = 'adam'
        dcec.compile(loss=[ 'kld', 'mse' ], loss_weights=[0.1, 1 ], optimizer=optimizer)
        dcec.fit(partitionsX, y=partitionsY, tol=0.001, maxiter=2e4,
                 update_interval=140,
                 save_dir='/home/dimitris/Desktop',
                 cae_weights=None)
        y_pred = dcec.y_pred

        #####################################################################
        autoencoder , encoder , decoder= baseline_model()
        partitionsX =partitionsX.reshape(1,-1)
        partitionsY = partitionsY.reshape(1,-1)
        autoencoder.fit(partitionsX,partitionsY,
                        epochs=10,
                        batch_size=200,
                        shuffle=True,)

        #feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
        features = autoencoder.predict(partitionsX)
        #featuresX = encoder.predict(partitionsX)
        print('feature shape=', features.shape)

        # use features for clustering

        #km = KMeans(n_clusters=2)

        #features = np.reshape(features, newshape=(features.shape[ 0 ], -1))
        #pred = km.fit_predict(features)

        seed = 7
        numpy.random.seed(seed)
        #for idx, pCurLbl in enumerate(partition_labels):

            #kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

            #for train, test in kfold.split(partitionsX[idx],partitionsY[idx]):
        for idx, pCurLbl in enumerate(partition_labels):
            estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, epochs=1,
                                                                         verbose=0 )
            estimator.fit(np.array(partitionsX[idx]),np.array(partitionsY[idx]))
                    #scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
                    #print("%s: %.2f%%" % ("acc: ", scores))

            models.append(estimator)
            self._partitionsPerModel[ estimator ] = partitionsX[idx]
        #model=stackedModels(models)
        # Update private models
        #models=[]
        #models.append(model)
        self._models = models

        # Return list of models
        return models, numpy.empty, numpy.empty , None


    def createSegmentsofTrData(self,trainX,trainY,timeStep,step):
        segs=[]
        labels=[]
        for i in range(0,len(trainX) - timeStep,step):
            xs = trainX[i:i+timeStep]
            ys = trainY[i:i+timeStep]
            segs.append(xs)
            labels.append(ys)
        return np.asarray(segs),np.asarray(labels)

    def createModelsForF(self, partitionsX, partitionsY, partition_labels):

        dataX =partitionsX
        dataY =partitionsY
        #dataY_t = np.concatenate(partition_labels)
        from numpy import array
        from numpy import argmax

        # define example
        #data = [ 1, 3, 2, 0, 3, 2, 2, 1, 0, 1 ]
        #data = array(data)
        #print(data)
        # one hot encode
        encodedX =[]
        #for x in dataX:
            #encoded =keras.utils.to_categorical(x)
            #encodedX.append(encoded)

        decodedY_inp=[]
        #for y in dataY:
            #encoded =keras.utils.to_categorical(y)
            #decodedY_inp.append(encoded)

        decodedY_trg = [ ]
        #for y in dataY_t:
            #encoded = keras.utils.to_categorical(y)
            #decodedY_trg.append(encoded)

        #print(encoded)
        # invert encoding
        #inverted = argmax(encoded[ 0 ])
        #print(inverted)
        #V = dataX[:,0]
        #V_n=dataX[:,1]

        #V = V.reshape((1, V.shape[ 0 ],1))
        #V_n = V_n.reshape((1, V_n.shape[ 0 ], 1))
        input_vectors =dataX
        target_vectors = dataY
        input_nums = set()
        target_nums = set()
        dataY=dataY.reshape(-1, 1)
        for x in dataX :
            for num in x:
                #if num not in input_nums:
                    input_nums.add(num)
        for y in dataY :
            for num in y:
                #if num not in target_nums:
                    target_nums.add(num)
        input_nums = list(input_nums)
        target_nums = list(target_nums)

        num_encoder_tokens = len(input_nums)
        num_decoder_tokens = len(target_nums)

        target_vectors = target_vectors.reshape(-1,1)
        max_encoder_seq_length = max([ len(num) for num in input_vectors ])
        max_decoder_seq_length = max([ len(num) for num in target_vectors ])

        input_token_index = dict(
            [ (num, i) for i, num in enumerate(input_nums) ])
        target_token_index = dict(
            [ (num, i) for i, num in enumerate(target_nums) ])

        #encoder_input_data = np.zeros(
            #(len(input_vectors), max_encoder_seq_length, num_encoder_tokens),
            #dtype='float32')

        encoder_input_data = np.zeros(
         (len(input_vectors), max_encoder_seq_length, num_encoder_tokens),
         dtype='float32')

        decoder_input_data = np.zeros(
            (len(target_vectors), max_decoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(target_vectors), max_decoder_seq_length,num_decoder_tokens),
            dtype='float32')

        for i, (input_vec, target_vec) in enumerate(zip(input_vectors, target_vectors)):
            for t, num in enumerate(input_vec):
                try:
                    encoder_input_data[ i, t, input_token_index[ num ] ] = dataX[i]
                except:
                    print(t)
            for t, num in enumerate(target_vec):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[ i, t, target_token_index[ num ] ] = dataX[i]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[ i, t - 1, target_token_index[ num ] ] = dataY[i]


        #dataX = dataX.reshape((1, dataX.shape[ 0 ], dataX.shape[ 1 ]))
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}

        def baseline_model():
            # create model
           #SplineWeights = SplinesCoef(partitionsX[idx],partitionsY[idx])
            #SplineWeights=np.resize(SplineWeights, (3, 10))
            #SplineWeights=np.array([np.resize(SplineWeights[0],10),np.resize(SplineWeights[1],10),np.resize(SplineWeights[2],10)])


            #model.load_weights()
            #weights=[SplineWeights,SplineWeights[0,0:]]
            #model.add(keras.layers.Dense(10, input_dim=2,activation='relu'))

            #input =keras.Input(shape=(partitionsX[0].shape[0],2)[1:2])
            #encoded = keras.layers.Dense(1000,input_dim=2, activation='relu')(input)
            #encoded = keras.layers.Dense(500, activation='relu')(encoded)
            #encoded = keras.layers.Dense(50, activation='relu')(encoded)

            #decoded = keras.layers.Dense(500, activation='relu')(encoded)
            #decoded = keras.layers.Dense(1000, activation='relu')(decoded)
            #decoded = keras.layers.Dense(1, activation='relu')(decoded)

            #model = keras.models.Model(input, decoded)
            #model.add(MyLayer(10))
            #num_encoder_tokens=1
            latent_dim=1
            #num_decoder_tokens=1

            encoder_inputs = keras.layers.Input(shape=(None,num_encoder_tokens ))
            encoder = keras.layers.LSTM(latent_dim, return_state=True)
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            # We discard `encoder_outputs` and only keep the states.
            encoder_states = [ state_h, state_c ]

            # Set up the decoder, using `encoder_states` as initial state.
            decoder_inputs = keras.layers.Input(shape=(None,num_decoder_tokens))
            # We set up our decoder to return full output sequences,
            # and to return internal states as well. We don't use the
            # return states in the training model, but we will use them in inference.
            decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                                 initial_state=encoder_states)
            decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)

            # Define the model that will turn
            # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
            model = keras.models.Model([ encoder_inputs, decoder_inputs ], decoder_outputs)

            #model = keras.models.Sequential()

            #model.add(keras.layers.GRU(100, input_shape=(dataX.shape[ 1 ], dataX.shape[2])))
            #model.add(keras.layers.Dense(50,activation='tanh', recurrent_activation='hard_sigmoid'))
            #model.add(keras.layers.Dense(10,activation='tanh', recurrent_activation='hard_sigmoid'))
            #model.add(keras.layers.Dense(1,activation='tanh', recurrent_activation='hard_sigmoid'))
            #model.layers[ 0 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])
            #model.layers[ 1 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])
            #model.layers[ 2 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])
            #model.layers[ 3 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])

            # Compile model
            #model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adagrad())

            model.compile(optimizer=keras.optimizers.Adagrad(), loss='mse')
            model.fit([ encoder_input_data, decoder_input_data ], decoder_target_data,
                         batch_size=64,
                         epochs=60,
                         validation_split=0.3,
                         )


            encoder_model = keras.models.Model(encoder_inputs, encoder_states)
            decoder_state_input_h =keras.layers.Input(shape=(latent_dim,))
            decoder_state_input_c =keras.layers.Input(shape=(latent_dim,))
            decoder_states_inputs = [ decoder_state_input_h, decoder_state_input_c ]
            decoder_outputs, state_h, state_c = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [ state_h, state_c ]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_model = keras.models.Model([ decoder_inputs ] + decoder_states_inputs,
                [ decoder_outputs ] + decoder_states)

            # Reverse-lookup token index to decode sequences back to
            # something readable.
            reverse_input_num_index = dict(
                (i, num) for num, i in input_token_index.items())
            reverse_target_num_index = dict(
                (i, num) for num, i in target_token_index.items())

            return model,encoder_model, decoder_model , reverse_input_num_index,reverse_target_num_index

        seed = 7
        numpy.random.seed(seed)
        #for idx, pCurLbl in enumerate(partition_labels):

            #kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

            #for train, test in kfold.split(partitionsX[idx],partitionsY[idx]):
        #for idx, pCurLbl in enumerate(partition_labels):
        genModel,encoder_model,decoder_model,reverse_input_num_index,reverse_target_num_index =\
        baseline_model()
            #keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, epochs=30)
        self.genModel=genModel
        self.encoder_model=encoder_model
        self.decoder_model=decoder_model
        self.reverse_input_num_index=reverse_input_num_index
        self.reverse_target_num_index=reverse_target_num_index
        self.num_decoder_tokens=num_decoder_tokens
        self.target_token_index=target_token_index

        def decode_sequence(input_seq,max_decoder_seq_length):
            # Encode the input as state vectors.

            states_value =encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            # Populate the first character of target sequence with the start character.
            #target_seq[ 0, 0, target_token_index[ '\t' ] ] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = [ ]
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict(
                    [ target_seq ] + states_value)
                v=genModel.predict([input_seq,output_tokens])
                # Sample a token
                sampled_token_indexv = np.argmax(v[ 0, -1, : ])
                sampled_numv = reverse_target_num_index[ sampled_token_indexv]

                sampled_token_index = np.argmax(output_tokens[ 0, -1, : ])
                sampled_num = reverse_target_num_index[ sampled_token_index ]

                decoded_sentence.append(sampled_num)

                # Exit condition: either hit max length
                # or find stop character.
                if (len(decoded_sentence) > max_decoder_seq_length):
                    stop_condition = True

                # Update the target sequence (of length 1).
                #target_seq = np.zeros((1, 1, num_decoder_tokens))
                #target_seq[ 0, 0, sampled_token_index ] = 1.

                # Update states
                states_value = [ h, c ]

            return decoded_sentence

        #for i in range(0,len(encodedX)):
            #genModel.fit([ np.array(encodedX[i]).reshape(21,encodedX[i].shape[1],1), decodedY_inp[i].reshape(21,decodedY_inp[i].shape[1],1) ], np.array(decodedY_trg[i]).reshape(21,decodedY_trg[i].shape[1],1),
            #batch_size = 1,
            #epochs = 10,
            #validation_split = 0.2)
            #genModel.fit(np.array(dataX),np.array(dataY))
                    #scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
                    #print("%s: %.2f%%" % ("acc: ", scores))


        #for idx, pCurLbl in enumerate(partition_labels):
            #genModel = keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, epochs=1,
                                                                  #verbose=0)
            #partitionsX[ idx ] = partitionsX[ idx ].reshape((partitionsX[ idx ].shape[ 0 ], 1, partitionsX[ idx ].shape[ 1 ]))
            #genModel.fit(np.array(partitionsX[ idx ]), np.array(partitionsY[ idx ]))
            # scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
            # print("%s: %.2f%%" % ("acc: ", scores))

            #models.append(genModel)
        self._partitionsPerModel[ genModel ] = dataX
        models.append(genModel)
        #model=stackedModels(models)
        # Update private models
        #models=[]
        #models.append(model)
        self._models = models
        self._decode_seq=decode_sequence

        # Return list of models
        return models, numpy.empty, numpy.empty , None

    def getBestPartitionForPoint(self, point, partitions):
        # mBest = None
        mBest = None
        dBestFit = 0
        # For each model
        for m in range(0, len(partitions)):
            # If it is a better for the point
            dCurFit = self.getFitnessOfPoint(partitions, m, point)
            if dCurFit > dBestFit:
                # Update the selected best model and corresponding fit
                dBestFit = dCurFit
                mBest = m

        if mBest == None:
            return 0, 0
        else:
            return mBest, dBestFit

    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):

        models = [ ]
        self.ClustersNum=len(partitionsX)
        self.modelId = -1
        partition_labels = partitionsX
        # Init model to partition map
        self._partitionsPerModel = {}
        def SplinesCoef(partitionsX, partitionsY):

           model= sp.Earth(use_fast=True)
           model.fit(partitionsX,partitionsY)

           return model.coef_

        class ClusteringLayer(keras.layers.Layer):
            """
            Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
            sample belonging to each cluster. The probability is calculated with student's t-distribution.
            # Example
            ```
                model.add(ClusteringLayer(n_clusters=10))
            ```
            # Arguments
                n_clusters: number of clusters.
                weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
                alpha: parameter in Student's t-distribution. Default to 1.0.
            # Input shape
                2D tensor with shape: `(n_samples, n_features)`.
            # Output shape
                2D tensor with shape: `(n_samples, n_clusters)`.
            """

            def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
                if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                    kwargs[ 'input_shape' ] = (kwargs.pop('input_dim'),)
                super(ClusteringLayer, self).__init__(**kwargs)
                self.n_clusters = n_clusters
                self.alpha = alpha
                self.initial_weights = weights
                self.input_spec = keras.layers.InputSpec(ndim=2)

            def build(self, input_shape):
                assert len(input_shape) == 2
                input_dim = input_shape[ 1 ]
                self.input_spec =keras.layers.InputSpec(dtype=keras.backend.floatx(), shape=(None, input_dim))
                #self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),name='clusters')
                self.clusters = self.add_weight(shape=(self.n_clusters, 1), name='clusters')
                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights
                self.built = True

            def call(self, inputs, **kwargs):
                """ student t-distribution, as same as used in t-SNE algorithm.
                         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
                Arguments:
                    inputs: the variable containing data, shape=(n_samples, n_features)
                Return:
                    q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
                """
                q = 1.0 / (1.0 + (keras.backend.sum(keras.backend.square(keras.backend.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
                q **= (self.alpha + 1.0) / 2.0
                q = keras.backend.transpose(keras.backend.transpose(q) / keras.backend.sum(q, axis=1))
                return q


            def compute_output_shape(self, input_shape):
                assert input_shape and len(input_shape) == 2

                return input_shape[ 0 ], self.n_clusters


            def get_config(self):
                config = {'n_clusters': self.n_clusters}
                base_config = super(ClusteringLayer, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

        class MyLayer(tf.keras.layers.Layer):


            def __init__(self, output_dim, **kwargs):
                self.output_dim = output_dim
                super(MyLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                # Create a trainable weight variable for this layer.
                #self.kernel = self.add_weight(name='kernel',
                                              #shape=(input_shape[1].value, self.output_dim),
                                              #initializer='uniform',
                                              #trainable=True)
                super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

            def call(self, inputs,modelId={'args':self.modelId}):

                x = inputs

                models = {"data": [ ]}
                intercepts = [ ]
                for csvM in csvModels:
                    id = csvM.split("_")[ 1 ]
                    piecewiseFunc = [ ]

                    with open(csvM) as csv_file:
                        data = csv.reader(csv_file, delimiter=',')
                        for row in data:
                            # for d in row:
                            if [ w for w in row if w == "Basis" ].__len__() > 0:
                                continue
                            if [ w for w in row if w == "(Intercept)" ].__len__() > 0:
                                intercepts.append(float(row[ 1 ]))
                                continue
                            if row.__len__() == 0:
                                continue
                            d = row[ 0 ]
                            if d.split("*").__len__() == 1:
                                split = ""
                                try:
                                    split = d.split('-')[ 0 ][ 2 ]
                                    if split != "x":
                                        num = float(d.split('-')[ 0 ].split('h(')[ 1 ])
                                        piecewiseFunc.append(
                                            tf.math.multiply(tf.cast(tf.math.greater(inputs, num), tf.float32),
                                                             float(row[ 1 ]) * (inputs - num)))
                                        if id == modelId['args'] or id == -1:
                                            inputs=tf.where(x >= num , float(row[ 1 ]) * (inputs - num), inputs)
                                    else:
                                        num = float(d.split('-')[ 1 ].split(')')[ 0 ])
                                        piecewiseFunc.append(tf.math.multiply(tf.cast(tf.math.less(inputs, num), tf.float32),
                                                                              float(row[ 1 ]) * (num - inputs)))
                                        if id == modelId['args'] or id == -1:
                                            inputs = tf.where(x >= num, float(row[ 1 ]) * (num - inputs), inputs)
                                except:
                                    piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                          float(row[ 1 ]) * (inputs)))
                                    if id == modelId['args'] or id == -1:
                                        inputs = tf.where(x >= 0, float(row[ 1 ]) * inputs, inputs)
                                    # continue

                            else:
                                funcs = d.split("*")
                                nums = [ ]
                                for r in funcs:
                                    try:
                                        if r.split('-')[ 0 ][ 2 ] != "x":
                                            nums.append(float(r.split('-')[ 0 ].split('h(')[ 1 ]))

                                        else:
                                            nums.append(float(r.split('-')[ 1 ].split(')')[ 0 ]))
                                        piecewiseFunc.append(tf.math.multiply(tf.cast(
                                            tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                                                                tf.math.greater(nums[ 1 ], x)), tf.float32),
                                                                              float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                                                      inputs - nums[ 1 ])))
                                        if id==modelId['args'] or id == -1:
                                            inputs = tf.where(x < nums[0] and x >= nums[0], float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                    inputs - nums[ 1 ]), inputs)
                                    except:
                                        try:
                                            if d.split('-')[ 0 ][ 2 ]=="x":
                                                piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                              float(row[ 1 ]) * (inputs) *(inputs - nums[0])))
                                                if id == modelId[ 'args' ] or id == -1:
                                                    inputs = tf.where(inputs >= nums[ 0 ] ,float(row[ 1 ]) * (inputs) * (inputs - nums[ 0 ]), x)

                                            else:
                                                piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                  float(row[ 1 ]) * (inputs) * (
                                                                                          nums[ 0 ] - inputs)))
                                                if id == modelId['args'] or id == -1:
                                                    inputs = tf.where(x < nums[ 0 ],
                                                                  float(row[ 1 ]) * (inputs) * ( nums[ 0 ]- inputs), inputs)
                                        except:
                                            piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                  float(row[ 1 ]) * (inputs) ))
                                            if id == modelId['args'] or id == -1:
                                                inputs = tf.where(x >= 0, float(row[ 1 ]) * (inputs),inputs)
                        model = {}
                        model[ "id" ] = id
                        model[ "funcs" ] = piecewiseFunc
                        models[ "data" ].append(model)

                modelId = 0 if modelId['args'] == -1 else modelId['args']

                # interc = tf.cast(x, tf.float32) + intercepts[self.modelId]
                #funcs = [ x for x in models[ 'data' ] if x[ 'id' ] == str(modelId) ][ 0 ][ 'funcs' ]
                #for f in funcs:
                    #inputs = f

                SelectedFuncs = intercepts[ modelId ] + np.sum(
                    [ x for x in models[ 'data' ] if x[ 'id' ] == str(modelId) ][ 0 ][ 'funcs' ])

                return inputs

            def get_config(self):
                config = {'sharp': float(self.sharp)}
                base_config = super(MyLayer, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.output_dim)

        def getFitnessOfPoint( partitions, cluster, point):
            return 1.0 / (1.0 + numpy.linalg.norm(np.mean(partitions[cluster]) - point))

        class ClassifyLayer(tf.keras.layers.Layer):

            def __init__(self, output_dim, **kwargs):
                self.output_dim = output_dim
                super(ClassifyLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                # Create a trainable weight variable for this layer.
                self.kernel = self.add_weight(name='kernel',
                                              shape=(input_shape[1], self.output_dim),
                                              initializer='uniform',
                                              trainable=True)
                super(ClassifyLayer, self).build(input_shape)  # Be sure to call this at the end



            def call(self, inputs):

                orig = inputs


                a = tf.where(orig <= 0.0, tf.zeros_like(inputs), inputs)
                b = tf.where(orig > 8.76,
                                  sr.coef_[0][0] * (inputs - 8.76), inputs)
                c = tf.where(orig < 8.76,
                                  sr.coef_[0][1] * (8.76 - inputs), inputs)

                d = tf.where(orig > 1.32,
                                  sr.coef_[0][2] * (inputs - 1.32), inputs)

                e = tf.where(tf.math.logical_and(tf.less(orig, 1.32), tf.greater(orig, 0)),
                                  (sr.coef_[0][3] * (1.32 - inputs)), inputs)

                return  keras.backend.sum(a,b,c,d,e)



            def get_config(self):
                config = {'sharp': float(self.sharp)}
                base_config = super(MyLayer, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.output_dim)

        def stackedModels(members):
            for i,pCurLbl in  enumerate(partition_labels):
                model = members[ i ]
                for layer in model.model.layers:
                    # make not trainable
                    layer.trainable = False
                    # rename to avoid 'unique layer name' issue
                    #layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
                # define multi-headed input
            ensemble_visible = [ model.model.input for model in members ]
            # concatenate merge output from each model
            ensemble_outputs = [ model.model.output for model in members ]
            merge = keras.layers.concatenate(ensemble_outputs)
            hidden = keras.layers.Dense(10, activation='relu')(merge)
            output = keras.layers.Dense(1, activation='relu')(hidden)
            model = keras.models.Model(inputs=ensemble_visible, outputs=output)
            # plot graph of ensemble
            #keras.utils.plot_model(model, show_shapes=True, to_file='/home/dimitris/model_graph.png')
            model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adagrad())
            return model

        def custom_activation3(inputs):

            x = inputs

            cond1 = tf.cast(tf.math.greater(x, 8.06), tf.float32)
            cond2 = tf.cast(tf.math.less_equal(x, 8.06), tf.float32)
            cond3 = tf.cast(tf.math.greater(x, 4.16014), tf.float32)
            cond4 = tf.cast(tf.math.less(x, 4.16014), tf.float32)
            cond5 = tf.cast(tf.math.greater(x, 18.597), tf.float32)
            cond6 = tf.cast(tf.math.less(x, 18.597), tf.float32)
            cond7 = tf.cast(tf.math.greater(x, 12.19), tf.float32)
            cond8 = tf.cast(tf.math.less(x, 12.19), tf.float32)
            cond9 = tf.cast(tf.math.greater(x, 5.1834), tf.float32)
            cond10 = tf.cast(tf.math.less(x, 5.1834), tf.float32)
            cond11 = tf.cast(tf.math.greater(x, 2.56829), tf.float32)
            cond12 = tf.cast(tf.math.less(x, 2.56829), tf.float32)
            cond11 = tf.cast(tf.math.greater(x, 6.13714), tf.float32)
            cond12 = tf.cast(tf.math.less(x, 6.13714), tf.float32)
            cond13 = tf.cast(tf.math.greater(x, 3.2), tf.float32)
            cond14 = tf.cast(tf.math.less(x, 3.2), tf.float32)
            cond15 = tf.cast(tf.math.greater(x,17.6784), tf.float32)
            cond16 = tf.cast(tf.math.less(x, 17.6784), tf.float32)
            #cond4 = tf.cast(tf.math.logical_or(tf.greater(x, 8.76), tf.less(x, 2.56)), tf.float32)

            intercept = sr.coef_[0][0]
            a = tf.math.multiply(cond1, sr.coef_[0][1] * (x - 8.06))
            b = tf.math.multiply(cond2, sr.coef_[0][2] * (8.06 - x))
            c = tf.math.multiply(cond3, sr.coef_[0][3] * (x -  4.16014))
            d = tf.math.multiply(cond4, sr.coef_[0][4] * (  4.16014 - x))
            e = tf.math.multiply(cond5, sr.coef_[0][5] * (x -  18.597))
            f = tf.math.multiply(cond6, sr.coef_[0][6] * ( 18.597 - x))
            g = tf.math.multiply(cond7, (sr.coef_[0][7] * ( 12.19 - x)))
            h = tf.math.multiply(cond8, (sr.coef_[0][8] *  (x -12.19 )))
            i = tf.math.multiply(cond7, (sr.coef_[ 0 ][ 9 ] * (5.1834 - x)))
            j = tf.math.multiply(cond8, (sr.coef_[ 0 ][ 10 ] * (x - 5.1834)))
            k = tf.math.multiply(cond7, (sr.coef_[ 0 ][ 7 ] * (2.56829- x)))
            l = tf.math.multiply(cond8, (sr.coef_[ 0 ][ 8 ] * (x - 2.56829)))
            m = tf.math.multiply(cond7, (sr.coef_[ 0 ][ 7 ] * (6.13714 - x)))
            n = tf.math.multiply(cond8, (sr.coef_[ 0 ][ 8 ] * (x -6.13714)))
            o = tf.math.multiply(cond7, (sr.coef_[ 0 ][ 7 ] * (3.2 - x)))
            p = tf.math.multiply(cond8, (sr.coef_[ 0 ][ 8 ] * (x - 3.2)))
            q = tf.math.multiply(cond7, (sr.coef_[ 0 ][ 7 ] * (17.6784 - x)))
            r = tf.math.multiply(cond8, (sr.coef_[ 0 ][ 8 ] * (x - 17.6784)))

            f = intercept + a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r

            return f

        def custom_activation2(inputs):

            x = inputs

            models = {"data": [ ]}
            intercepts = [ ]
            for csvM in csvModels:
                id = csvM.split("_")[ 1 ]
                id =  0 if id=='Gen'  else int(id)
                piecewiseFunc = [ ]

                with open(csvM) as csv_file:
                    data = csv.reader(csv_file, delimiter=',')
                    for row in data:
                        # for d in row:
                        if [ w for w in row if w == "Basis" ].__len__() > 0:
                            continue
                        if [ w for w in row if w == "(Intercept)" ].__len__() > 0:
                            intercepts.append(float(row[ 1 ]))
                            continue
                        if row.__len__() == 0:
                            continue
                        d = row[ 0 ]
                        coeffS = 1
                        #float(row[1])
                        if d.split("*").__len__() == 1:
                            split = ""
                            try:
                                split = d.split('-')[ 0 ][ 2 ]
                                if split != "x":
                                    num = float(d.split('-')[ 0 ].split('h(')[ 1 ])

                                    #if coeffS < 10:
                                    piecewiseFunc.append(
                                        tf.math.multiply(tf.cast(tf.math.greater(x, num), tf.float32),
                                                         coeffS * (x - num)))

                                    #if id ==  self.modelId:
                                        #inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                                else:
                                    num = float(d.split('-')[ 1 ].split(')')[ 0 ])
                                    #if coeffS < 10:
                                    piecewiseFunc.append(
                                        tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                                         coeffS * (num - x)))

                                    #if id == self.modelId:
                                        #inputs = tf.where(x >= num, float(row[ 1 ]) * (num - inputs), inputs)
                            except:
                                #if coeffS < 10:
                                piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                          coeffS * (x)))
                                #if id ==  self.modelId:
                                    #inputs = tf.where(x >= 0, float(row[ 1 ]) * inputs, inputs)
                                # continue

                        else:
                            funcs = d.split("*")
                            nums = [ ]
                            for r in funcs:
                                try:
                                    if r.split('-')[ 0 ][ 2 ] != "x":
                                        nums.append(float(r.split('-')[ 0 ].split('h(')[ 1 ]))

                                    else:
                                        nums.append(float(r.split('-')[ 1 ].split(')')[ 0 ]))
                                    piecewiseFunc.append(tf.math.multiply(tf.cast(
                                        tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                                                            tf.math.greater(nums[ 1 ], x)), tf.float32),
                                        float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                inputs - nums[ 1 ])))
                                    if id ==  self.modelId:
                                        inputs = tf.where(x < nums[ 0 ] and x >= nums[ 0 ],
                                                          float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                                  inputs - nums[ 1 ]), inputs)
                                except:
                                    try:
                                        if d.split('-')[ 0 ][ 2 ] == "x":
                                            piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                  float(row[ 1 ]) * (inputs) * (
                                                                                          inputs - nums[ 0 ])))
                                            if id ==  self.modelId :
                                                inputs = tf.where(inputs >= nums[ 0 ],
                                                                  float(row[ 1 ]) * (inputs) * (inputs - nums[ 0 ]), x)

                                        else:
                                            piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                  float(row[ 1 ]) * (inputs) * (
                                                                                          nums[ 0 ] - inputs)))
                                            if id == self.modelId:
                                                inputs = tf.where(x < nums[ 0 ],
                                                                  float(row[ 1 ]) * (inputs) * (nums[ 0 ] - inputs),
                                                                  inputs)
                                    except:
                                        piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                              float(row[ 1 ]) * (inputs)))
                                        if id == self.modelId :
                                            inputs = tf.where(x >= 0, float(row[ 1 ]) * (inputs), inputs)
                    model = {}
                    model[ "id" ] = id
                    model[ "funcs" ] = piecewiseFunc
                    models[ "data" ].append(model)

            #modelId = 0 if modelId[ 'args' ] == -1 else modelId[ 'args' ]

            # interc = tf.cast(x, tf.float32) + intercepts[self.modelId]
            # funcs = [ x for x in models[ 'data' ] if x[ 'id' ] == str(modelId) ][ 0 ][ 'funcs' ]
            # for f in funcs:
            # inputs = f
            SelectedFuncs=0
            #from keras.layers import Input, Lambda
            #out = Lambda(lambda a: a[0] + a[1])([piecewiseFunc])
            SelectedFuncs= tf.math.multiply(tf.cast(x, tf.float32),x) if ([ x for x in models[ 'data' ] if x[ 'id' ] == self.modelId ][ 0 ][ 'funcs' ]).__len__()==0 else np.sum([ x for x in models[ 'data' ] if x[ 'id' ] == self.modelId ][ 0 ][ 'funcs' ])
            #for f in [ x for x in models[ 'data' ] if x[ 'id' ] == self.modelId ][ 0 ][ 'funcs' ]:
                #SelectedFuncs+= f
                #added=keras.layers.Add()([f])
            #SelectedFuncs =  tf.keras.backend.sum(
                #[ x for x in models[ 'data' ] if x[ 'id' ] == self.modelId ][ 0 ][ 'funcs' ],keepdims=True)

            return  SelectedFuncs


        def custom_activation3(inputs):

            x = inputs

            cond1 = tf.cast(tf.math.greater(x, 8.76), tf.float32)
            cond2 = tf.cast(tf.math.less_equal(x, 8.76), tf.float32)
            cond3 = tf.cast(tf.math.greater(x, 18.597), tf.float32)
            cond4 = tf.cast(tf.math.less(x, 18.597), tf.float32)
            cond5 = tf.cast(tf.math.greater(x, 12.3237), tf.float32)
            cond6 = tf.cast(tf.math.less(x, 12.3237), tf.float32)
            cond7 = tf.cast(tf.math.greater(x, 17.6784), tf.float32)
            cond8 = tf.cast(tf.math.less(x, 17.6784), tf.float32)
            #cond4 = tf.cast(tf.math.logical_or(tf.greater(x, 8.76), tf.less(x, 2.56)), tf.float32)

            intercept = sr.coef_[0][0]
            a = tf.math.multiply(cond1, sr.coef_[0][1] * (x - 8.76))
            b = tf.math.multiply(cond2, sr.coef_[0][2] * (8.76 - x))
            c = tf.math.multiply(cond3, sr.coef_[0][3] * (x - 18.597))
            d = tf.math.multiply(cond4, sr.coef_[0][4] * ( 18.597 - x))
            e = tf.math.multiply(cond5, sr.coef_[0][5] * (x -  12.3237))
            f = tf.math.multiply(cond6, sr.coef_[0][6] * ( 12.3237 - x))
            g = tf.math.multiply(cond7, (sr.coef_[0][7] * ( 17.6784 - x)))
            h = tf.math.multiply(cond8, (sr.coef_[0][8] *  (x - 17.6784 )))

            f = intercept + a + b + c + d + e + f + g + h

            return f

        def custom_activation(inputs):

            x = inputs

            models = {"data": [ ]}
            intercepts = [ ]
            for csvM in csvModels:
                id = csvM.split("_")[ 1 ]
                piecewiseFunc = [ ]

                with open(csvM) as csv_file:
                    data = csv.reader(csv_file, delimiter=',')
                    for row in data:
                        # for d in row:
                        if [ w for w in row if w == "Basis" ].__len__() > 0:
                            continue
                        if [ w for w in row if w == "(Intercept)" ].__len__() > 0:
                            intercepts.append(float(row[ 1 ]))
                            continue
                        if row.__len__() == 0:
                            continue
                        d = row[ 0 ]
                        if d.split("*").__len__() == 1:
                            split = ""
                            try:
                                split = d.split('-')[ 0 ][ 2 ]
                                if split != "x":
                                    num = float(d.split('-')[ 0 ].split('h(')[ 1 ])
                                    piecewiseFunc.append(
                                        tf.math.multiply(tf.cast(tf.math.greater(x, num), tf.float32),
                                                         float(row[ 1 ]) * (inputs - num)))
                                    if id == self.modelId:
                                        inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                                else:
                                    num = float(d.split('-')[ 1 ].split(')')[ 0 ])
                                    piecewiseFunc.append(
                                        tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                                                         float(row[ 1 ]) * (num - inputs)))
                                    if id == self.modelId:
                                        inputs = tf.where(x >= num, float(row[ 1 ]) * (num - inputs), inputs)
                            except:
                                piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                      float(row[ 1 ]) * (inputs)))
                                if id == self.modelId:
                                    inputs = tf.where(x >= 0, float(row[ 1 ]) * inputs, inputs)
                                # continue

                        else:
                            funcs = d.split("*")
                            nums = [ ]
                            for r in funcs:
                                try:
                                    if r.split('-')[ 0 ][ 2 ] != "x":
                                        nums.append(float(r.split('-')[ 0 ].split('h(')[ 1 ]))

                                    else:
                                        nums.append(float(r.split('-')[ 1 ].split(')')[ 0 ]))
                                    piecewiseFunc.append(tf.math.multiply(tf.cast(
                                        tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                                                            tf.math.greater(nums[ 1 ], x)), tf.float32),
                                        float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                inputs - nums[ 1 ])))
                                    if id == self.modelId:
                                        inputs = tf.where(x < nums[ 0 ] and x >= nums[ 0 ],
                                                          float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                                                                  inputs - nums[ 1 ]), inputs)
                                except:
                                    try:
                                        if d.split('-')[ 0 ][ 2 ] == "x":
                                            piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                  float(row[ 1 ]) * (inputs) * (
                                                                                          inputs - nums[ 0 ])))
                                            if id == self.modelId:
                                                inputs = tf.where(inputs >= nums[ 0 ],
                                                                  float(row[ 1 ]) * (inputs) * (inputs - nums[ 0 ]), x)

                                        else:
                                            piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                                  float(row[ 1 ]) * (inputs) * (
                                                                                          nums[ 0 ] - inputs)))
                                            if id == self.modelId:
                                                inputs = tf.where(x < nums[ 0 ],
                                                                  float(row[ 1 ]) * (inputs) * (nums[ 0 ] - inputs),
                                                                  inputs)
                                    except:
                                        piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                                                                              float(row[ 1 ]) * (inputs)))
                                        if id == self.modelId:
                                            inputs = tf.where(x >= 0, float(row[ 1 ]) * (inputs), inputs)
                    model = {}
                    model[ "id" ] = id
                    model[ "funcs" ] = piecewiseFunc
                    models[ "data" ].append(model)

            # modelId = 0 if modelId[ 'args' ] == -1 else modelId[ 'args' ]

            # interc = tf.cast(x, tf.float32) + intercepts[self.modelId]
            # funcs = [ x for x in models[ 'data' ] if x[ 'id' ] == str(modelId) ][ 0 ][ 'funcs' ]
            # for f in funcs:
            # inputs = f

            # SelectedFuncs = intercepts[ self.modelId ] + np.sum(
            # [ x for x in models[ 'data' ] if x[ 'id' ] == str( self.modelId) ][ 0 ][ 'funcs' ])

            return inputs

            #cond1 = tf.cast(tf.math.greater(x, 8.76), tf.float32)
            #cond2 = tf.cast(tf.math.less_equal(x, 8.76), tf.float32)
            #cond3 = tf.cast(tf.math.logical_and(tf.less(x, 8.76), tf.greater(x, 2.56)), tf.float32)
            #cond4 = tf.cast(tf.math.logical_or(tf.greater(x, 8.76), tf.less(x, 2.56)), tf.float32)


            #intercept = sr.coef_[ 0 ][ 0 ]
            #a = tf.math.multiply(cond1, sr.coef_[ 0 ][ 1 ] * (x - 8.76))
            #b = tf.math.multiply(cond2, sr.coef_[ 0 ][ 2 ] * (8.76 - x))
            #c = tf.math.multiply(cond3, sr.coef_[ 0 ][ 3 ] * (x - 2.56) * (8.76 - x))
            #d = tf.math.multiply(cond4, (sr.coef_[ 0 ][ 4 ] * (2.56 - x)* (x - 8.76)))

            #f = intercept + a + b + c + d

            #return f

        def custom_activation1(inputs):

            x = inputs


            cond1 = tf.cast(tf.math.greater(x, 8.76), tf.float32)
            cond2 = tf.cast(tf.math.less_equal(x, 8.76), tf.float32)
            cond3 = tf.cast(tf.math.greater(x,1.32), tf.float32)
            cond4 = tf.cast(tf.math.less_equal(x, 1.32), tf.float32)

            intercept = sr.coef_[0][0]
            a = tf.math.multiply(cond1,   sr.coef_[0][1] * (x - 8.76))
            b = tf.math.multiply(cond2, sr.coef_[0][2] * (8.76 - x))
            c = tf.math.multiply(cond3,  sr.coef_[0][3] * (x - 1.32))
            d = tf.math.multiply(cond4, (sr.coef_[0][4] * (1.32 - x)))

            f =intercept+ a + b + c + d

            return f
                #keras.backend.sum(inputs , keepdims=True)

        def sum_output_shape(input_shapes):
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            assert shape1 == shape2  # else hadamard product isn't possible
            return [tuple(shape1), tuple(shape2[:-1])]

        def preTrainedWeights(candidatePoint):
            from scipy.spatial import Delaunay, ConvexHull
            import math
            from trianglesolver import solve, degree
            dataXnew = partitionsX
            dataYnew = partitionsY
            triNew = Delaunay(dataXnew)
            XpredN =0
            weights = []
            indexes = []
            for k in range(0, len(triNew.vertices)):
                # simplicesIndexes
                # k=triNew.find_simplex(dataXnew[i])
                V1 = dataXnew[triNew.vertices[k]][0]
                V2 = dataXnew[triNew.vertices[k]][1]
                V3 = dataXnew[triNew.vertices[k]][2]

                # plt.plot(candidatePoint[ 0 ], candidatePoint[ 1 ], 'o', markersize=8)
                # plt.show()
                x = 1
                b2 = triNew.transform[ k, :2 ].dot(candidatePoint - triNew.transform[ k, 2 ])
                W1 = b2[ 0 ]
                W2 = b2[ 1 ]
                W3 = 1 - np.sum(b2)
                if (W1 == 0 and W2 == 0) or (W3 == 0 and W1==0) or (W2==0 and W3==0) :

                    a = spatial.distance.euclidean(V1, V2)
                    b = spatial.distance.euclidean(V1, V3)
                    c = spatial.distance.euclidean(V3, V2)
                    a, b, c, A, B, C = solve(b=a, c=b, a=c)
                    A = A / degree
                    B = B / degree
                    C = C / degree
                    W1 = math.sin(A)
                    W2 = math.sin(B)
                    W3 = math.sin(C)

                    if 1 == 1:

                        rpm1 = dataYnew[triNew.vertices[k]][0]
                        rpm2 = dataYnew[triNew.vertices[k]][1]
                        rpm3 = dataYnew[triNew.vertices[k]][2]

                        x = 0

                        ##barycenters (W12,W23,W13) of neighboring triangles and solutions of linear 3x3 sustem in order to find gammas.


                        #############################
                        ##end of barycenters
                        flgExc = False
                        try:

                            neighboringVertices1 = []

                            for u in range(0, 2):
                                try:
                                    neighboringVertices1.append(triNew.vertex_neighbor_vertices[1][
                                                                triNew.vertex_neighbor_vertices[0][
                                                                    triNew.vertices[k][u]]:
                                                                triNew.vertex_neighbor_vertices[0][
                                                                    triNew.vertices[k][u] + 1]])
                                except:
                                    break
                            neighboringTri = triNew.vertices[
                                triNew.find_simplex(dataXnew[np.concatenate(np.array(neighboringVertices1))])]
                            if (1 == 1):

                                nRpms = []
                                nGammas = []

                                rpms = []
                                for s in neighboringTri:
                                    V1n = dataXnew[s][0]
                                    V2n = dataXnew[s][1]
                                    V3n = dataXnew[s][2]

                                    rpm1n = dataYnew[s][0]
                                    rpm2n = dataYnew[s][1]
                                    rpm3n = dataYnew[s][2]

                                    rpms.append([rpm1n, rpm2n, rpm3n])
                                    ###barycentric coords of neighboring points in relation to initial triangle
                                    eq1 = np.array([[(V1[0] - V3[0]), (V2[0] - V3[0])],
                                                    [(V1[1] - V3[1]), (V2[1] - V3[1])]])

                                    eq2 = np.array([V1n[0] - V3[0], V1n[1] - V3[1]])
                                    solutions = np.linalg.solve(eq1, eq2)

                                    W1n = solutions[0]
                                    W2n = solutions[1]
                                    W3n = 1 - solutions[0] - solutions[1]

                                    B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                    nRpms.append(rpm1n - B1)

                                    nGammas.append(np.array([2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n]))
                                    ####################################

                                    eq1 = np.array([[(V1[0] - V3[0]), (V2[0] - V3[0])],
                                                    [(V1[1] - V3[1]), (V2[1] - V3[1])]])

                                    eq2 = np.array([V2n[0] - V3[0], V2n[1] - V3[1]])
                                    solutions = np.linalg.solve(eq1, eq2)

                                    W1n = solutions[0]
                                    W2n = solutions[1]
                                    W3n = 1 - solutions[0] - solutions[1]

                                    B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                    nRpms.append(rpm2n - B1)

                                    nGammas.append(np.array([2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n]))
                                    ##################################################

                                    eq1 = np.array([[(V1[0] - V3[0]), (V2[0] - V3[0])],
                                                    [(V1[1] - V3[1]), (V2[1] - V3[1])]])

                                    eq2 = np.array([V3n[0] - V3[0], V3n[1] - V3[1]])
                                    solutions = np.linalg.solve(eq1, eq2)

                                    W1n = solutions[0]
                                    W2n = solutions[1]
                                    W3n = 1 - solutions[0] - solutions[1]

                                    B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                    nRpms.append(rpm3n - B1)
                                    nGammas.append(np.array([2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n]))

                                nGammas = np.array(nGammas)
                                nRpms = np.array(nRpms)
                                from sklearn.linear_model import LinearRegression
                                lr = LinearRegression()
                                leastSqApprx = lr.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
                                XpredN = ((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) +
                                          2 * W1 * W2 * leastSqApprx.coef_[0][0] +
                                          2 * W2 * W3 * leastSqApprx.coef_[0][1] +
                                          2 * W1 * W3 * leastSqApprx.coef_[0][2])
                                weights.append(leastSqApprx.coef_[0][0])
                                weights.append(leastSqApprx.coef_[0][1])
                                weights.append(leastSqApprx.coef_[0][2])
                                # weights.append(np.mean(leastSqApprx.coef_))

                        except:
                            # print(str(e))
                            x = 1
                    return XpredN

        def baseline_modelDeepCl():
            #create model
            model = keras.models.Sequential()


            model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            #model.add(keras.layers.Dense(len(partition_labels) * 2, input_shape=(2,)))
            #model.add(keras.layers.Dense(len(partition_labels) * 3, input_shape=(2,)))
            #model.add(keras.layers.Dense(len(partition_labels) * 2, input_shape=(2,)))
            #model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,)))
            #model.add(keras.layers.Dense(10, input_shape=(2,)))
            #model.add(keras.layers.Dense(5, input_shape=(2,)))
            model.add(keras.layers.Activation(custom_activation2))
            model.add(keras.layers.Dense(1,)) #activation=custom_activation
            #model.add(keras.layers.Activation(custom_activation2))

            #model.add(keras.layers.Activation(custom_activation2))
            #
            #model.add(keras.layers.Activation(custom_activation))
            #model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss='mse' , optimizer=keras.optimizers.Adam())
            return model

        def baseline_model():
            #create model
            model = keras.models.Sequential()

            #model.add(keras.layers.Dense(len(partition_labels)*2, input_shape=(2,)))
            #model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,) ))
            #model.add(keras.layers.Activation(custom_activation2))

            #len(DeepClpartitionLabels)
            model.add(keras.layers.Dense(len(DeepClpartitionLabels), input_shape=(2,)))
            #model.add(MyLayer(genModelKnots))
            #model.add(keras.layers.Dense(15, input_shape=(2,)))

            #model.add(keras.layers.Dense(genModelKnots, input_shape=(2,)))
            #model.add(keras.layers.Dense(2, input_shape=(2,)))
            #model.add(keras.layers.Activation(custom_activation2(inputs=model.layers[2].output, modelId=1)))
            model.add(keras.layers.Activation(custom_activation2))
            model.add(keras.layers.Dense(1,input_shape=(2,))) #activation=custom_activation
            #model.add(keras.layers.Activation(custom_activation))
            #model.add(keras.layers.Activation('linear'))  # activation=custom_activation
            # Compile model
            model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            return model


        seed = 7
        numpy.random.seed(seed)


        partitionsX=X
        partitionsY=Y
        #weights = preTrainedWeights()

        dims = [ 2 ,1000,500,200 , 100, len(partition_labels) ]
        init = keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in',
                                                  distribution='uniform')
        # pretrain_optimizer =keras.optimizers.SGD(lr=1, momentum=0.9)
        seed = 7
        numpy.random.seed(seed)


        pretrain_epochs = 3
        batch_size = 100

        #autoencoder, encoder = self.autoencoder(dims, init=init)
        partitionsX.reshape(-1,2)
        #dataUpdatedX = np.append(partitionsX, np.asmatrix([partitionsY]).T, axis=1)
        #autoencoder.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        #autoencoder.fit(partitionsX, partitionsX, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
        #autoencoder.fit(partitionsX, partitionsX, batch_size=batch_size,
                        #epochs=3)
        # autoencoder.save_weights(save_dir + '/ae_weights.h5')

        #clustering_layer = ClusteringLayer(len(partition_labels), name='clustering')(encoder.output)
        #model1 = keras.models.Model(inputs=encoder.input, outputs=clustering_layer)

        #model1.compile(optimizer=keras.optimizers.Adam(), loss='kld')
        #SGD(0.01, 0.9)
        dataUpdatedX = np.append(partitionsX, np.asmatrix([ partitionsY ]).T, axis=1)
        kmeans = KMeans(n_clusters=len(partition_labels), n_init=20)
        #dataUpdatedX = partitionsX.reshape(-1, 2)
        y_predDeepCl = kmeans.fit_predict(partitionsX)
        #cl_centers=np.array([np.mean(k) for k in kmeans.cluster_centers_]).reshape(-1,1)
        #model1.get_layer(name='clustering').set_weights([ cl_centers ])

        #q = model1.predict(partitionsX, verbose=0)
        #y_predDeepCl = q.argmax(1)

        sModel=[]
        sr = sp.Earth()
        sr.fit(partitionsX,partitionsY)
        sModel.append(sr)
        import csv
        csvModels = [ ]
        genModelKnots=[]
        self.modelId = 0
        self.triRpm = 0
        self.count= 0
        for models in sModel:
            modelSummary = str(models.summary()).split("\n")[ 4: ]

            with open('./model_Gen_.csv', mode='w') as data:
                csvModels.append('./model_Gen_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    [ 'Basis', 'Coeff' ])
                for row in modelSummary:
                    row = np.delete(np.array(row.split(" ")), [ i for i, x in enumerate(row.split(" ")) if x == "" ])
                    try:
                        basis = row[ 0 ]
                        pruned = row[ 1 ]
                        coeff = row[ 2 ]

                        if pruned == "No":
                            data_writer.writerow([ basis, coeff ])
                            genModelKnots.append(basis)
                    except:
                        x = 0

            genModelKnots = len(genModelKnots)
            #modelCount += 1
            #models.append(autoencoder)

        ########SET K MEANS INITIAL WEIGHTS TO CLUSTERING LAYER
        def customLoss1(yTrue, yPred):
            self.triRpm = preTrainedWeights(partitionsX[self.count])
            self.count += 1
            return (tf.losses.mean_squared_error(yTrue,(yPred + self.triRpm)/2 ))

        def customLoss(yTrue, yPred):
            return tf.losses.categorical_crossentropy(yTrue,yPred) +  tf.losses.kullback_leibler_divergence(yTrue,yPred)
                     #+ tf.losses.kullback_leibler_divergence(yTrue,yPred))
        #
        ###############
        #X_train, X_test, y_train, y_test = train_test_split(partitionsX, partitionsY, test_size=0.33, random_state=seed)
        #estimatorD = baseline_modelDeepCl()
        #dataUpdatedX = np.append(partitionsX, np.asmatrix([partitionsY]).T, axis=1)
        #estimatorD.fit(partitionsX, partitionsY, epochs=30)

        #model2 = keras.models.Model(inputs=estimatorD.input, outputs=estimatorD.layers[-2].output)

        #model2.compile(optimizer=keras.optimizers.Adam(),loss=customLoss )
        #model2.fit(partitionsX,partitionsY,epochs=50)

        #q = model2.predict(partitionsX)
        #y_predDeepCl = q.argmax(1)

        #pred =np.sum(model2.predict(partitionsX[0].reshape(1,2), verbose=0))/15

        DeepCLpartitionsX = [ ]
        DeepCLpartitionsY = [ ]
        DeepClpartitionLabels = [ ]
        # For each label
        x2 = partitionsX.reshape(-1, 2)
        y2 = partitionsY.reshape(-1, 1)
        for curLbl in np.unique(y_predDeepCl):
            # Create a partition for X using records with corresponding label equal to the current
            if np.asarray(x2[ y_predDeepCl == curLbl ]).__len__() < 10:
                continue
            DeepCLpartitionsX.append(np.asarray(x2[ y_predDeepCl == curLbl ]))
            # Create a partition for Y using records with corresponding label equal to the current
            DeepCLpartitionsY.append(np.asarray(y2[ y_predDeepCl == curLbl ]))
            # Keep partition label to ascertain same order of results
            DeepClpartitionLabels.append(curLbl)

        srModels = []
        for idx, pCurLbl in enumerate(DeepClpartitionLabels):
            srM = sp.Earth()
            srM.fit(np.array(DeepCLpartitionsX[idx]), np.array(DeepCLpartitionsY[idx]))
            srModels.append(srM)
        modelCount =1
        import csv
        #csvModels = []
        ClModels={"data":[]}

        for models in srModels:
            modelSummary = str(models.summary()).split("\n")[4:]
            basisM = [ ]
            with open('./model_'+str(modelCount)+'_.csv', mode='w') as data:
                csvModels.append('./model_'+str(modelCount)+'_.csv')
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                ['Basis', 'Coeff'])
                for row in modelSummary:
                    row=np.delete(np.array(row.split(" ")), [i for i, x in enumerate(row.split(" ")) if x == ""])
                    try:
                        basis = row[0]
                        pruned = row[1]
                        coeff = row[2]
                        if pruned == "No":
                            data_writer.writerow([basis, coeff])
                            basisM.append(basis)
                    except:
                        x=0
                model = {}
                model[ "id" ] = modelCount
                model[ "funcs" ] = len(basisM)
                ClModels[ "data" ].append(model)
            modelCount+=1
        estimator = baseline_model()
        #for i in range(0,len(partitionsX)):
            #self.triRpm = preTrainedWeights(partitionsX[i])
        #checkpoint =keras.callbacks.ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                                     #save_best_only=True, mode='auto', period=1) callbacks=[checkpoint]
        estimator.fit(partitionsX, partitionsY,epochs=100 ,validation_split=0.33)

         # validation_data=(X_test,y_test)

        def insert_intermediate_layer_in_keras(model,layer_id, new_layer):


            layers = [ l for l in model.layers ]

            x = layers[ 0 ].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)
                if i == len(layers)-1:
                    x = keras.layers.Dense(1)(x)
                else:
                    x = layers[ i ](x)
            #x = new_layer(x)
            new_model = keras.Model(inputs=model.input, outputs=x)
            #new_model.add(new_layer)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            #.add(x)
            return new_model

        def replace_intermediate_layer_in_keras(model, layer_id,layer_id1 ,new_layer,new_layer1):

            layers = [ l for l in model.layers ]

            x = layers[ 0 ].output
            for i in range(1, len(layers)):
                if i == layer_id:
                    x = new_layer(x)
                elif  i == layer_id1:
                    x = new_layer1(x)
                else:
                    x = layers[ i ](x)

            new_model = keras.Model(inputs=model.input, outputs=x)
            new_model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            return new_model

        #for train, test in kfold.split(partitionsX[idx],partitionsY[idx]):
        #if len(partition_labels)>0:
        #models.append(estimator)
        #models={}
        #models[ 300 ] = estimator

        NNmodels=[]
        for idx, pCurLbl in enumerate(DeepClpartitionLabels):
                #partitionsX[ idx ]=partitionsX[idx].reshape(-1,2)

                #estimator = baseline_model()
                self.modelId = idx + 1
                modelId=idx
                #estimatorCl = baseline_model()
                #if idx==0:
                    #estimator.add(keras.layers.Activation(custom_activation2))
                #else:
                numOfNeurons = [x for x in ClModels['data'] if x['id']==idx+1][0]['funcs']
                #estimatorCl=replace_intermediate_layer_in_keras(estimator, -1 ,MyLayer(5))
                #estimatorCl = insert_intermediate_layer_in_keras(estimator, 1, keras.layers.Dense(numOfNeurons))
                if numOfNeurons > 1 :
                    estimatorCl = replace_intermediate_layer_in_keras(estimator, 0,1, keras.layers.Dense(len(DeepClpartitionLabels)+ numOfNeurons),keras.layers.Activation(custom_activation2))
                else:
                    estimatorCl = replace_intermediate_layer_in_keras(estimator, 0,1, keras.layers.Dense(len(DeepClpartitionLabels)+ numOfNeurons),keras.layers.Activation(custom_activation2))
                #estimatorCl = insert_intermediate_layer_in_keras(estimator, 1, MyLayer(numOfNeurons))
                #estimatorCl = insert_intermediate_layer_in_keras(estimator,0,keras.layers.Activation(custom_activation2))
                #estimatorCl.add(keras.layers.Activation(custom_activation2))
                #estimator.compile()
                #estimator.layers[3] = custom_activation2(inputs=estimator.layers[2].output, modelId=idx) if idx ==0 else estimator.layers[3]
                #estimator.layers[3] = custom_activation2 if idx ==3 else estimator.layers[3]
                estimatorCl.fit(np.array(DeepCLpartitionsX[idx]),np.array(DeepCLpartitionsY[idx]),epochs=100)
                    #scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
                    #print("%s: %.2f%%" % ("acc: ", scores))
                NNmodels.append(estimatorCl)
                #models[pCurLbl]=estimator
                #self._partitionsPerModel[ estimator ] = partitionsX[idx]
        # Update private models
        #models=[]
        #NNmodels.append(estimator)

        #NNmodels.append(estimator)
        self._models = NNmodels

        # Return list of models
        return estimator, kmeans ,numpy.empty, numpy.empty , estimator , DeepCLpartitionsX

    def createModelsForConv(self,partitionsX, partitionsY, partition_labels):
        ##Conv1D NN
        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}

        TIME_PERIODS=len(partitionsX[0])
        STEP_DISTANCE = 40
        x_train, y_train = self.createSegmentsofTrData(partitionsX[ 0 ], partitionsY[ 0 ], TIME_PERIODS, STEP_DISTANCE)

        num_time_periods, num_sensors =partitionsX[0].shape[0], partitionsX[0].shape[1]
        num_classes = 1
        input_shape = (num_sensors)
        #x_train = x_train.reshape(x_train.shape[ 0 ], input_shape)


        model_m = keras.models.Sequential()
        model_m.add(keras.layers.Reshape((TIME_PERIODS,num_sensors, ), input_shape=(num_sensors,)))
        model_m.add(keras.layers.Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
        model_m.add(keras.layers.Conv1D(100, 10, activation='relu'))
        model_m.add(keras.layers.MaxPooling1D(2))
        model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
        model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
        model_m.add(keras.layers.GlobalAveragePooling1D())
        model_m.add(keras.layers.Dropout(0.3))
        model_m.add(keras.layers.Dense(num_classes, activation='softmax'))
        print(model_m.summary())

        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]

        model_m.compile(loss='mean_squared_error',
                        optimizer='adam', metrics=[ 'accuracy' ])

        BATCH_SIZE = 400
        EPOCHS = 50
        for idx, pCurLbl in enumerate(partition_labels):
           #for i in range(0,x_train.shape[1]):
           history = model_m.fit(np.nan_to_num(partitionsX[idx]),np.nan_to_num(partitionsY[idx]),
                                      epochs=EPOCHS,
                                     )

        models.append(model_m)
        self._partitionsPerModel[ model_m ] = partitionsX[ idx ]

        # Update private models

        self._models = models

        # Return list of models
        return models, numpy.empty, numpy.empty


    def createModelsForS(self,partitionsX, partitionsY, partition_labels):

        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        #curModel = keras.models.Sequential()

        # This will add a fully connected neural network layer with 32#neurons, each#taking#13#inputs, and with activation function ReLU
        #curModel.add(keras.layers.Dense(1, input_dim=3, activation='relu'))

        #curModel.compile(loss='mean_squared_error',
        #                 optimizer='sgd',
        #                 metrics=[ 'mae' ])
        # Fit to data

        #input_A = keras.layers.Input(shape=partitionsX[0].shape)
        #input_B = keras.layers.Input(shape=partitionsX[1].shape)
        #input_C = keras.layers.Input(shape=partitionsX[2].shape)
        #A = keras.layers.Dense(1)(input_A)
        #A = keras.layers.Dense(1)(A)
        #B = keras.layers.concatenate([ input_A, input_B, input_C ], mode='concat')
        #B = keras.layers.Dense(1)(B)
        #C = keras.layers.concatenate([ A, B ], mode='concat')
        #C = keras.layers.Dense(1)(C)
        #B = keras.layers.concatenate([ A, B ], mode='concat')
        #B = keras.layers.Dense(1)(B)
        #A = keras.layers.Dense(1)(A)

        #curModel.fit(np.asarray([ [[0,2], [0,2] ],[[0,2], [0,2] ], [[0,2], [0,2] ] ]),
                     #np.asarray([ [[0,2], [0,2] ],[[0,2], [0,2] ], [[0,2], [0,2] ] ]), epochs=10)
        #curModel.fit(partitionsX, partitionsY, epochs=10)

        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = keras.models.Sequential()
            partitionsX[ idx ] = np.reshape(partitionsX[ idx ], (partitionsX[ idx ].shape[ 0 ], 1, partitionsX[ idx ].shape[ 1 ]))

            # This will add a fully connected neural network layer with 32#neurons, each#taking#13#inputs, and with activation function ReLU
            curModel.add(keras.layers.Conv1D(2,input_shape=partitionsX[idx].shape[1:],activation='relu'))
            #curModel.add(keras.layers.LSTM(3, input_shape=partitionsX[ idx ].shape[ 1: ], activation='relu'))
            #curModel.add(keras.layers.Flatten())
            curModel.add(keras.layers.Dense(1))

            ##optimizers
            adam=keras.optimizers.Adam(lr=0.001)
            rmsprop=keras.optimizers.RMSprop(lr=0.01)
            adagrad=keras.optimizers.Adagrad(lr=0.01)
            sgd=keras.optimizers.SGD(lr=0.001)
            ######

            curModel.compile(loss='mean_squared_error',
              optimizer=adam,metrics=['mae'])
            # Fit to data
            curModel.fit(np.nan_to_num(partitionsX[idx]),np.nan_to_num(partitionsY[idx]), epochs=100)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[curModel] = partitionsX[idx]

        # Update private models
        self._models = models

        # Return list of models
        return models , numpy.empty ,numpy.empty


    def createModelsForX1(self, partitionsX, partitionsY, partition_labels):

        SAVE_PATH = './save'
        EPOCHS = 1
        LEARNING_RATE = 0.001
        MODEL_NAME = 'test'

        model = sp.Earth(use_fast=True)
        model.fit(partitionsX[0],partitionsY[0])
        W_1= model.coef_

        self._partitionsPerModel = {}


        ######################################################
        # Data specific constants
        n_input = 784  # data input
        n_classes = 1  #  total classes
        # Hyperparameters
        max_epochs = 10
        learning_rate = 0.5
        batch_size = 10
        seed = 0
        n_hidden = 3  # Number of neurons in the hidden layer


        # Gradient Descent optimization  for updating weights and biases
        # Execution Graph
        c_t=[]
        c_test=[]
        models=[]

        for idx, pCurLbl in enumerate(partition_labels):
            n_input = partitionsX[ idx ].shape[ 1 ]
            xs = tf.placeholder(tf.float32, [ None, n_input ])
            ys = tf.placeholder("float")

            output = self.initNN(xs, 2)
            cost = tf.reduce_mean(tf.square(output - ys))
            # our mean square error cost function
            train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
            ##

            init_op = tf.global_variables_initializer()
            #name_to_var_map = {var.op.name: var for var in tf.global_variables()}

            #name_to_var_map[ "loss/total_loss/avg" ] = name_to_var_map[
            #    "conv_classifier/loss/total_loss/avg" ]
            #del name_to_var_map[ "conv_classifier/loss/total_loss/avg" ]

            saver = tf.train.Saver()
            if idx==1:
                with tf.Session() as sess:
                    # Initiate session and initialize all vaiables
                        #################################################################
                        # output = self.initNN(xs,W_1,2)
                        sess.run(init_op)

                        for i in range(EPOCHS):
                      #for j in range(len(partitionsX[idx].shape[ 0 ]):
                          for j in range(0,len(partitionsX[idx])):

                             all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                             tf.variables_initializer(all_variables)
                             sess.run([ cost, train ], feed_dict={xs: [partitionsX[idx][ j, : ]], ys: [partitionsY[idx][ j ]]})
                        # Run cost and train with each sample
                        #c_t.append(sess.run(cost, feed_dict={xs: partitionsX[idx], ys: partitionsY[idx]}))
                        #c_test.append(sess.run(cost, feed_dict={xs: X_test, ys: y_test}))
                        #print('Epoch :', i, 'Cost :', c_t[ i ])
                        models.append(sess)
                        #self._partitionsPerModel[sess]  = partitionsX[idx]

                        self._models = models
                        saver.save(sess, SAVE_PATH + '/' + MODEL_NAME+'_'+str(idx) + '.ckpt')
                sess.close()
        return  models , xs , output

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))


class TensorFlow(BasePartitionModeler):


    def initNN(self,X_data,input_dim):
        #X=np.concatenate(partitionsX)
        W_1 = tf.Variable(tf.random_uniform([ input_dim, 10 ]))
        #W_1 = np.array(W_1).reshape(-1,2)
        #weightsShape = len(W_1[0])
        #W_1=tf.Variable(np.float32(W_1[0]))
        b_1 = tf.Variable(tf.zeros([ 10 ]))
        layer_1 = tf.add(tf.matmul(X_data,W_1),b_1)
        #layer_1 = tf.add( W_1, b_1)
        layer_1 = tf.nn.relu(layer_1)
        # layer 1 multiplying and adding bias then activation function

        W_2 = tf.Variable(tf.random_uniform([ 10,10 ]))
        b_2 = tf.Variable(tf.zeros([10 ]))
        layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
        layer_2 = tf.nn.relu(layer_2)
        # layer 2 multiplying and adding bias then activation function

        W_3 = tf.Variable(tf.random_uniform([10,10 ]))
        b_3 = tf.Variable(tf.zeros([ 10 ]))
        layer_3 = tf.add(tf.matmul(layer_2, W_3), b_3)
        layer_3 = tf.nn.relu(layer_3)
        #layer 3 multiplying and adding bias then activation function
        W_O = tf.Variable(tf.random_uniform([10, 1 ]))
        b_O = tf.Variable(tf.zeros([ 1 ]))
        output = tf.add(tf.matmul(layer_1, W_O), b_O)


        return  output
        # O/p layer multiplying and adding bias then activation function
        # notice output layer has one node only since performing #regression
        #return output

    def initNN_2(self,x,weights,biases):
        # Hidden layer with RELU activation
        h_layer_1 = tf.add(tf.matmul(x, weights[ 'h1' ]), biases[ 'h1' ])
        out_layer_1 = tf.sigmoid(h_layer_1)
        # Output layer with linear activation
        h_out = tf.matmul(out_layer_1, weights[ 'out' ]) + biases[ 'out' ]

        return h_out

    def createModelsForAUTO_ENC(self, partitionsX, partitionsY, partition_labels):
        x = partitionsX
        y = partitionsY

        #tri = Delaunay(x.reshape(-1,2))
        # x = x.reshape((x.shape[ 0 ], -1))
        n_clusters = 15
        # kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
        # y_pred_kmeans = kmeans.fit_predict(x)
        x = x[ 0, : ]
        x = x.reshape((x.shape[ 0 ], -1))
        ###########
        dims = [ x.shape[-1], 1000, 500, 200 , 100 ,n_clusters]
        init = keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in',
                                                  distribution='uniform')
        # pretrain_optimizer =keras.optimizers.SGD(lr=1, momentum=0.9)
        seed = 7
        numpy.random.seed(seed)

        # x=x.reshape(1,-1)
        # y=y.reshape(1,-1)
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)

        pretrain_epochs = 2
        batch_size = 100
        save_dir = '/home/dimitris/Desktop/results'
        autoencoder, encoder =self.autoencoder(dims, init=init)

        autoencoder.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
        #autoencoder.save_weights(save_dir + '/ae_weights.h5')

        #autoencoder.load_weights(save_dir + '/ae_weights.h5')

        return encoder.get_weights()[encoder.weights.__len__() - 1]

    def autoencoder(self,dims, act='relu', init='glorot_uniform'):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = len(dims) - 1
        # input
        input_img = keras.layers.Input(shape=(dims[ 0 ],), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(n_stacks - 1):
            x = keras.layers.Dense(dims[ i + 1 ], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
            # r=keras.layers.Reshape((dims[i+1],1,))(x)
            # lstm_x = keras.layers.LSTM(10, return_sequences=True, return_state=True)(r)
            # lstm_x_r =keras.layers.Reshape((dims[i+1],))(lstm_x)
        # hidden layer
        # encoded = keras.layers.Dense(dims[ -1 ], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(lstm_x[2])
        encoded = keras.layers.Dense(dims[ -1 ], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
        # hidden layer, features are extracted from here

        x = encoded

        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            x = keras.layers.Dense(dims[ i ], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = keras.layers.Dense(dims[ 0 ], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        return keras.models.Model(inputs=input_img, outputs=decoded, name='AE'), keras.models.Model(inputs=input_img,
                                                                                                    outputs=encoded,
                                                                                                    name='encoder')

    def createModelsForAUTO(self, partitionsX, partitionsY, partition_labels):

        models = [ ]
        # Init model to partition map
        #partitionsX = np.concatenate(partitionsX)
        #partitionsY = np.concatenate(partitionsY)

        self._partitionsPerModel = {}
        def SplinesCoef(partitionsX, partitionsY):

           model= sp.Earth(use_fast=True)
           model.fit(partitionsX,partitionsY)

           return model.coef_

        class MyLayer(tf.keras.layers.Layer):

            def __init__(self, output_dim):
                self.output_dim = output_dim
                super(MyLayer, self).__init__()

            def build(self, input_shape):

                # Create a trainable weight variable for this layer.
                self.kernel = self.add_weight(name='kernel',
                                              shape=(input_shape[ 1 ], self.output_dim),
                                              trainable=True)
                super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

            def call(self, x,mask=None):
                return keras.backend.dot(x, self.kernel)

            def compute_output_shape(self, input_shape):
                return (input_shape[ 0 ], self.output_dim)

        class DCEC(object):
            def __init__(self,
                         input_shape=(partitionsX.shape[0],partitionsX.shape[1], 1),
                         filters=[ 32, 64, 128, 10 ],
                         n_clusters=10,
                         alpha=1.0):

                super(DCEC, self).__init__()

                self.n_clusters = n_clusters
                self.input_shape = input_shape
                self.alpha = alpha
                self.pretrained = False
                self.y_pred = [ ]

                self.cae = CAE(input_shape, filters)
                hidden = self.cae.get_layer(name='embedding').output
                self.encoder =keras.models.Model(inputs=self.cae.input, outputs=hidden)

                # Define DCEC model
                clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)

                self.model = keras.models.Model(inputs=self.cae.input,
                                   outputs=[ clustering_layer, self.cae.output ])

            def pretrain(self, x, batch_size=256, epochs=10, optimizer='adam', save_dir='results/temp'):
                print('...Pretraining...')
                self.cae.compile(optimizer=optimizer, loss='mse')

                csv_logger =keras.callbacks.CSVLogger('/home/dimitris/Desktop' + '/pretrain_log.csv')

                # begin training
                t0 = time()
                # callbacks=[ csv_logger ]
                x=x[0:400]

                #self.cae.fit(x, x, batch_size=batch_size, epochs=epochs)
                print('Pretraining time: ', time() - t0)
                self.cae.save(save_dir + '/pretrain_cae_model.h5')
                print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
                self.pretrained = True

            def load_weights(self, weights_path):
                self.model.load_weights(weights_path)

            def extract_feature(self, x):  # extract features from before clustering layer
                return self.encoder.predict(x)

            def predict(self, x):
                q, _ = self.model.predict(x, verbose=0)
                return q.argmax(1)

            @staticmethod
            def target_distribution(q):
                weight = q ** 2 / q.sum(0)
                return (weight.T / weight.sum(1)).T

            def compile(self, loss=[ 'kld', 'mse' ], loss_weights=[ 1, 1 ], optimizer='adam'):
                self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

            def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
                    update_interval=140, cae_weights=None, save_dir='./results/temp'):

                print('Update interval', update_interval)
                save_interval = x.shape[ 0 ] / batch_size * 5
                print('Save interval', save_interval)

                # Step 1: pretrain if necessary
                t0 = time()
                if not self.pretrained and cae_weights is None:
                    print('...pretraining CAE using default hyper-parameters:')
                    print('   optimizer=\'adam\';   epochs=200')
                    self.pretrain(x, batch_size, save_dir=save_dir)
                    self.pretrained = True
                elif cae_weights is not None:
                    self.cae.load_weights(cae_weights)
                    print('cae_weights is loaded successfully.')

                # Step 2: initialize cluster centers using k-means
                t1 = time()
                print('Initializing cluster centers with k-means.')
                kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
                self.y_pred = kmeans.fit_predict(self.encoder.predict(x[ 0:400 ].reshape(1, 400, 1, 1))[ 0 ].reshape(-1,1))
                #self.y_pred = kmeans.fit_predict(self.encoder.predict(x[0,400].reshape(1,400,1,1)))
                y_pred_last = np.copy(self.y_pred)
                self.model.get_layer(name='clustering').set_weights([ kmeans.cluster_centers_ ])

                # Step 3: deep clustering
                # logging file
                import csv, os
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                logfile = open(save_dir + '/dcec_log.csv', 'w')
                logwriter = csv.DictWriter(logfile, fieldnames=[ 'iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr' ])
                logwriter.writeheader()

                t2 = time()
                loss = [ 0, 0, 0 ]
                index = 0
                for ite in range(int(maxiter)):
                    if ite % update_interval == 0:
                        q, _ = self.model.predict(x[0:400].reshape(1,400,1,1), verbose=0)
                        p = self.target_distribution(q)  # update the auxiliary target distribution p

                        # evaluate the clustering performance
                        self.y_pred = q.argmax(1)
                        if y is not None:
                            acc = np.round(metrics.acc(y, self.y_pred), 5)
                            nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                            ari = np.round(metrics.ari(y, self.y_pred), 5)
                            loss = np.round(loss, 5)
                            logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[ 0 ], Lc=loss[ 1 ], Lr=loss[ 2 ])
                            logwriter.writerow(logdict)
                            print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                        # check stop criterion
                        delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[ 0 ]
                        y_pred_last = np.copy(self.y_pred)
                        if ite > 0 and delta_label < tol:
                            print('delta_label ', delta_label, '< tol ', tol)
                            print('Reached tolerance threshold. Stopping training.')
                            logfile.close()
                            break

                    # train on batch
                    if (index + 1) * batch_size > x.shape[ 0 ]:
                        loss = self.model.train_on_batch(x=x[ index * batch_size:: ],
                                                         y=[ p[ index * batch_size:: ], x[ index * batch_size:: ] ])
                        index = 0
                    else:
                        loss = self.model.train_on_batch(x=x[ index * batch_size:(index + 1) * batch_size ],
                                                         y=[ p[ index * batch_size:(index + 1) * batch_size ],
                                                             x[ index * batch_size:(index + 1) * batch_size ] ])
                        index += 1

                    # save intermediate model
                    if ite % save_interval == 0:
                        # save DCEC model checkpoints
                        print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                        #self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

                    ite += 1

                # save the trained model
                logfile.close()
                #print('saving model to:', save_dir + '/dcec_model_final.h5')
                #self.model.save_weights(save_dir + '/dcec_model_final.h5')
                t3 = time()
                print('Pretrain time:  ', t1 - t0)
                print('Clustering time:', t3 - t1)
                print('Total time:     ', t3 - t0)

        class ClusteringLayer(keras.layers.Layer):
            """
            Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
            sample belonging to each cluster. The probability is calculated with student's t-distribution.
            # Example
            ```
                model.add(ClusteringLayer(n_clusters=10))
            ```
            # Arguments
                n_clusters: number of clusters.
                weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
                alpha: parameter in Student's t-distribution. Default to 1.0.
            # Input shape
                2D tensor with shape: `(n_samples, n_features)`.
            # Output shape
                2D tensor with shape: `(n_samples, n_clusters)`.
            """

            def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
                if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                    kwargs[ 'input_shape' ] = (kwargs.pop('input_dim'),)
                super(ClusteringLayer, self).__init__(**kwargs)
                self.n_clusters = n_clusters
                self.alpha = alpha
                self.initial_weights = weights
                self.input_spec = keras.layers.InputSpec(ndim=2)

            def build(self, input_shape):
                assert len(input_shape) == 2
                input_dim = input_shape[ 1 ]
                self.input_spec =keras.layers.InputSpec(dtype=keras.backend.floatx(), shape=(None, input_dim))
                #self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),name='clusters')
                self.clusters = self.add_weight(shape=(self.n_clusters, 1), name='clusters')
                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights
                self.built = True

            def call(self, inputs, **kwargs):
                """ student t-distribution, as same as used in t-SNE algorithm.
                         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
                Arguments:
                    inputs: the variable containing data, shape=(n_samples, n_features)
                Return:
                    q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
                """
                q = 1.0 / (1.0 + (keras.backend.sum(keras.backend.square(keras.backend.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
                q **= (self.alpha + 1.0) / 2.0
                q = keras.backend.transpose(keras.backend.transpose(q) / keras.backend.sum(q, axis=1))
                return q

            def compute_output_shape(self, input_shape):
                assert input_shape and len(input_shape) == 2
                return input_shape[ 0 ], self.n_clusters

            def get_config(self):
                config = {'n_clusters': self.n_clusters}
                base_config = super(ClusteringLayer, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

        def stackedModels(members):
            for i,pCurLbl in  enumerate(partition_labels):
                model = members[ i ]
                for layer in model.model.layers:
                    # make not trainable
                    layer.trainable = False
                    # rename to avoid 'unique layer name' issue
                    #layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
                # define multi-headed input
            ensemble_visible = [ model.model.input for model in members ]
            # concatenate merge output from each model
            ensemble_outputs = [ model.model.output for model in members ]
            merge = keras.layers.concatenate(ensemble_outputs)
            hidden = keras.layers.Dense(10, activation='relu')(merge)
            output = keras.layers.Dense(1, activation='relu')(hidden)
            model = keras.models.Model(inputs=ensemble_visible, outputs=output)
            # plot graph of ensemble
            #keras.utils.plot_model(model, show_shapes=True, to_file='/home/dimitris/model_graph.png')
            #model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adagrad())
            return model


        def baseline_model1():
            model =keras.models.Sequential()

            # 1st convolution layer
            model.add(keras.layers.Conv2D(16, (3, 3)  # 16 is number of filters and (3, 3) is the size of the filter.
                             , padding='same', input_shape=(partitionsX.shape[0],1,1)))
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

            # 2nd convolution layer
            model.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

            # here compressed version

            # 3rd convolution layer
            model.add(keras.layers.Conv2D(2, (3, 3), padding='same'))  # apply 2 filters sized of (3x3)
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.UpSampling2D((2, 2)))

            # 4th convolution layer
            model.add(keras.layers.Conv2D(16, (3, 3), padding='same'))
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.UpSampling2D((2, 2)))

            model.add(keras.layers.Conv2D(1, (3, 3), padding='same'))
            model.add(keras.layers.Activation('sigmoid'))

            model.compile(optimizer='adadelta', loss='binary_crossentropy')

            return model


        def CAE(input_shape=(partitionsX[0:400].shape[0],partitionsX[0:400].shape[1], 1), filters=[ 32, 64, 128, 10 ]):
            model =keras.models.Sequential()
            if input_shape[ 0 ] % 8 == 0:
                pad3 = 'same'
            else:
                pad3 = 'valid'
            model.add(keras.layers.Conv2D(filters[ 0 ], 5, strides=2, padding='same', activation='relu', name='conv1',
                             input_shape=input_shape))

            model.add(keras.layers.Conv2D(filters[ 1 ], 5, strides=2, padding='same', activation='relu', name='conv2'))

            model.add(keras.layers.Conv2D(filters[ 2 ], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(units=filters[ 3 ], name='embedding'))
            model.add(
                keras.layers.Dense(units=filters[ 2 ] * int(input_shape[ 0 ] / 8) * int(input_shape[ 0 ] / 8), activation='relu'))

            model.add(keras.layers.Reshape((int(input_shape[ 0 ] / 8), 1, filters[ 2 ])))
            model.add(keras.layers.Conv2DTranspose(filters[ 1 ], 3, strides=1, padding=pad3, activation='relu', name='deconv3'))

            model.add(keras.layers.Conv2DTranspose(filters[ 0 ], 5, strides=1, padding='same', activation='relu', name='deconv2'))

            model.add(keras.layers.Conv2DTranspose(1, 1, strides=1, padding='same', name='deconv1'))
            #model.add(keras.layers.Reshape((int(input_shape[ 0 ] ), 1, 1)))
            model.summary()
            return model

        def baseline_model():
            # create model
           #SplineWeights = SplinesCoef(partitionsX[idx],partitionsY[idx])
            #SplineWeights=np.resize(SplineWeights, (3, 10))
            #SplineWeights=np.array([np.resize(SplineWeights[0],10),np.resize(SplineWeights[1],10),np.resize(SplineWeights[2],10)])
            # this is the size of our encoded representations
            encoding_dim = partitionsX.shape[0]/10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

            # this is our input placeholder
            input_img = keras.Input(shape=(partitionsX.shape[0],))
            # "encoded" is the encoded representation of the input
            encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_img)
            # "decoded" is the lossy reconstruction of the input
            decoded = keras.layers.Dense(partitionsX.shape[0], activation='sigmoid')(encoded)

            # this model maps an input to its reconstruction
            autoencoder = keras.models.Model(input_img, decoded)

            # this model maps an input to its encoded representation
            encoder = keras.models.Model(input_img, encoded)

            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = keras.Input(shape=(encoding_dim,))
            # retrieve the last layer of the autoencoder model
            decoder_layer = autoencoder.layers[ -1 ]
            # create the decoder model
            decoder = keras.models.Model(encoded_input, decoder_layer(encoded_input))

            # Compile model
            autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

            return autoencoder , encoder , decoder
        #################################################   clustering NN
        x=partitionsX
        y=partitionsY
        #x = x.reshape((x.shape[ 0 ], -1))
        n_clusters=self.ClustersNum
        #kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
        #y_pred_kmeans = kmeans.fit_predict(x)
        x = x[ 0, : ]
        x = x.reshape((x.shape[ 0 ], -1))
        ###########
        dims = [ x.shape[-1], 1000, 500, 200 , 100 ,n_clusters ]
        init =keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')
        #pretrain_optimizer =keras.optimizers.SGD(lr=1, momentum=0.9)
        seed = 7
        numpy.random.seed(seed)

        #x=x.reshape(1,-1)
        #y=y.reshape(1,-1)
        #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)

        pretrain_epochs = 2
        batch_size = 100
        save_dir = '/home/dimitris/Desktop/results'
        autoencoder, encoder =self.autoencoder(dims, init=init)


        #keras.utils.plot_model(autoencoder, to_file='/home/dimitris/Desktop/autoencoder.png', show_shapes=True)
        #from IPython.display import Image
        #Image(filename='/home/dimitris/Desktop/autoencoder.png')
        #x=x[:,0]
        #X_train=X_train.reshape(-1,1)
        #y_train = y_train.reshape(-1,1)
        #X_test = X_test.reshape(-1, 1)
        #y_test = y_test.reshape(-1, 1)
        #x = x.reshape(1, -1)
        #y = y.reshape(1, -1)
        autoencoder.compile(optimizer=keras.optimizers.Adagrad(), loss='mse')
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
        #autoencoder.save_weights(save_dir + '/ae_weights.h5')

        #autoencoder.load_weights(save_dir + '/ae_weights.h5')

        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)

        #modelLayer =  keras.layers.Input(shape=(1,),name='inputModel')
        model = keras.models.Model(inputs=encoder.input, outputs=clustering_layer)

        model.compile(optimizer=keras.optimizers.Adam(), loss='kld')

        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(encoder.predict(x)[0].reshape(-1,1))
        #y_pred = kmeans.fit_predict(x.reshape(-1, 1))

        CLpartitionsX = [ ]
        CLpartitionsY = [ ]
        partitionLabels = [ ]
        # For each label
        x1 = partitionsX.reshape(-1, 2)
        y1 = partitionsY.reshape(-1, 1)
        for curLbl in np.unique(y_pred):
            # Create a partition for X using records with corresponding label equal to the current
            CLpartitionsX.append(np.asarray(x1[ y_pred == curLbl ]))
            # Create a partition for Y using records with corresponding label equal to the current
            CLpartitionsY.append(np.asarray(y1[ y_pred == curLbl ]))
            # Keep partition label to ascertain same order of results
            partitionLabels.append(curLbl)

        ####retrain autoencoder with clustered data
        #return partitionsX,partitionsX,partitionLabels ,kmeans.cluster_centers_
        #for k in partitionLabels:
            #autoencoder.fit(partitionsX[k], partitionsY[k], batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)

        y_pred_last = np.copy(y_pred)
        ########SET K MEANS INITIAL WEIGHTS TO CLUSTERING LAYER
        model.get_layer(name='clustering').set_weights([ kmeans.cluster_centers_ ])

        loss = 0
        index = 0
        maxiter = 8000
        update_interval = 140
        index_array = np.arange(x.shape[ 0 ])
        tol = 0.001  # tolerance threshold to stop training
        y=None
        #x=x.reshape(1,-1)
        def target_distribution(q):
            weight = q ** 2 / q.sum(0)
            return (weight.T / weight.sum(1)).T
        #####start training
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = model.predict(x, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[ 0 ]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[ index * batch_size: min((index + 1) * batch_size, x.shape[ 0 ]) ]
            loss = model.train_on_batch(x=x[ idx ], y=p[ idx ])
            index = index + 1 if (index + 1) * batch_size <= x.shape[ 0 ] else 0
        ###########


        #model.save_weights(save_dir + '/DEC_model_final.h5')
        #model.load_weights(save_dir + '/DEC_model_final.h5')

        # Eval.
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p
        models.append(autoencoder)

        y_predDeepCl = q.argmax(1)

        DeepCLpartitionsX = [ ]
        DeepCLpartitionsY = [ ]
        DeepClpartitionLabels = [ ]
        # For each label
        x2 = partitionsX.reshape(-1, 2)
        y2 = partitionsY.reshape(-1, 1)
        for curLbl in np.unique(y_predDeepCl):
            # Create a partition for X using records with corresponding label equal to the current
            DeepCLpartitionsX.append(np.asarray(x2[ y_predDeepCl == curLbl ]))
            # Create a partition for Y using records with corresponding label equal to the current
            DeepCLpartitionsY.append(np.asarray(y2[ y_predDeepCl == curLbl ]))
            # Keep partition label to ascertain same order of results
            DeepClpartitionLabels.append(curLbl)

        self._models = models

        # Return list of models
        #return models, numpy.empty, numpy.empty, None
        return DeepCLpartitionsX, DeepCLpartitionsY, DeepClpartitionLabels
        # evaluate the clustering performance

        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)
        #########################################################################################
        ###########################################CONV2D##################
        # prepare the DCEC model
        dcec = DCEC(input_shape=(partitionsX[0:400].shape[0],partitionsX[0:400].shape[1], 1), filters=[ 32, 64, 128, 10 ], n_clusters=10)
        #plot_model(dcec.model, to_file=args.save_dir + '/dcec_model.png', show_shapes=True)
        dcec.model.summary()

        # begin clustering.
        optimizer = 'adam'
        dcec.compile(loss=[ 'kld', 'mse' ], loss_weights=[0.1, 1 ], optimizer=optimizer)
        dcec.fit(partitionsX, y=partitionsY, tol=0.001, maxiter=2e4,
                 update_interval=140,
                 save_dir='/home/dimitris/Desktop',
                 cae_weights=None)
        y_pred = dcec.y_pred

        #####################################################################
        autoencoder , encoder , decoder= baseline_model()
        partitionsX =partitionsX.reshape(1,-1)
        partitionsY = partitionsY.reshape(1,-1)
        autoencoder.fit(partitionsX,partitionsY,
                        epochs=10,
                        batch_size=200,
                        shuffle=True,)

        #feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
        features = autoencoder.predict(partitionsX)
        #featuresX = encoder.predict(partitionsX)
        print('feature shape=', features.shape)

        # use features for clustering

        #km = KMeans(n_clusters=2)

        #features = np.reshape(features, newshape=(features.shape[ 0 ], -1))
        #pred = km.fit_predict(features)

        seed = 7
        numpy.random.seed(seed)
        #for idx, pCurLbl in enumerate(partition_labels):

            #kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

            #for train, test in kfold.split(partitionsX[idx],partitionsY[idx]):
        for idx, pCurLbl in enumerate(partition_labels):
            estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, epochs=1,
                                                                         verbose=0 )
            estimator.fit(np.array(partitionsX[idx]),np.array(partitionsY[idx]))
                    #scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
                    #print("%s: %.2f%%" % ("acc: ", scores))

            models.append(estimator)
            self._partitionsPerModel[ estimator ] = partitionsX[idx]
        #model=stackedModels(models)
        # Update private models
        #models=[]
        #models.append(model)
        self._models = models

        # Return list of models
        return models, numpy.empty, numpy.empty , None

    def createSegmentsofTrData(self,trainX,trainY,timeStep,step):
        segs=[]
        labels=[]
        for i in range(0,len(trainX) - timeStep,step):
            xs = trainX[i:i+timeStep]
            ys = trainY[i:i+timeStep]
            segs.append(xs)
            labels.append(ys)
        return np.asarray(segs),np.asarray(labels)

    def createModelsForF(self, partitionsX, partitionsY, partition_labels):

        dataX =partitionsX
        dataY =partitionsY
        #dataY_t = np.concatenate(partition_labels)
        from numpy import array
        from numpy import argmax

        # define example
        #data = [ 1, 3, 2, 0, 3, 2, 2, 1, 0, 1 ]
        #data = array(data)
        #print(data)
        # one hot encode
        encodedX =[]
        #for x in dataX:
            #encoded =keras.utils.to_categorical(x)
            #encodedX.append(encoded)

        decodedY_inp=[]
        #for y in dataY:
            #encoded =keras.utils.to_categorical(y)
            #decodedY_inp.append(encoded)

        decodedY_trg = [ ]
        #for y in dataY_t:
            #encoded = keras.utils.to_categorical(y)
            #decodedY_trg.append(encoded)

        #print(encoded)
        # invert encoding
        #inverted = argmax(encoded[ 0 ])
        #print(inverted)
        #V = dataX[:,0]
        #V_n=dataX[:,1]

        #V = V.reshape((1, V.shape[ 0 ],1))
        #V_n = V_n.reshape((1, V_n.shape[ 0 ], 1))
        input_vectors =dataX
        target_vectors = dataY
        input_nums = set()
        target_nums = set()
        dataY=dataY.reshape(-1, 1)
        for x in dataX :
            for num in x:
                #if num not in input_nums:
                    input_nums.add(num)
        for y in dataY :
            for num in y:
                #if num not in target_nums:
                    target_nums.add(num)
        input_nums = list(input_nums)
        target_nums = list(target_nums)

        num_encoder_tokens = len(input_nums)
        num_decoder_tokens = len(target_nums)

        target_vectors = target_vectors.reshape(-1,1)
        max_encoder_seq_length = max([ len(num) for num in input_vectors ])
        max_decoder_seq_length = max([ len(num) for num in target_vectors ])

        input_token_index = dict(
            [ (num, i) for i, num in enumerate(input_nums) ])
        target_token_index = dict(
            [ (num, i) for i, num in enumerate(target_nums) ])

        #encoder_input_data = np.zeros(
            #(len(input_vectors), max_encoder_seq_length, num_encoder_tokens),
            #dtype='float32')

        encoder_input_data = np.zeros(
         (len(input_vectors), max_encoder_seq_length, num_encoder_tokens),
         dtype='float32')

        decoder_input_data = np.zeros(
            (len(target_vectors), max_decoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(target_vectors), max_decoder_seq_length,num_decoder_tokens),
            dtype='float32')

        for i, (input_vec, target_vec) in enumerate(zip(input_vectors, target_vectors)):
            for t, num in enumerate(input_vec):
                try:
                    encoder_input_data[ i, t, input_token_index[ num ] ] = dataX[i]
                except:
                    print(t)
            for t, num in enumerate(target_vec):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[ i, t, target_token_index[ num ] ] = dataX[i]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[ i, t - 1, target_token_index[ num ] ] = dataY[i]


        #dataX = dataX.reshape((1, dataX.shape[ 0 ], dataX.shape[ 1 ]))
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}

        def baseline_model():
            # create model
           #SplineWeights = SplinesCoef(partitionsX[idx],partitionsY[idx])
            #SplineWeights=np.resize(SplineWeights, (3, 10))
            #SplineWeights=np.array([np.resize(SplineWeights[0],10),np.resize(SplineWeights[1],10),np.resize(SplineWeights[2],10)])


            #model.load_weights()
            #weights=[SplineWeights,SplineWeights[0,0:]]
            #model.add(keras.layers.Dense(10, input_dim=2,activation='relu'))

            #input =keras.Input(shape=(partitionsX[0].shape[0],2)[1:2])
            #encoded = keras.layers.Dense(1000,input_dim=2, activation='relu')(input)
            #encoded = keras.layers.Dense(500, activation='relu')(encoded)
            #encoded = keras.layers.Dense(50, activation='relu')(encoded)

            #decoded = keras.layers.Dense(500, activation='relu')(encoded)
            #decoded = keras.layers.Dense(1000, activation='relu')(decoded)
            #decoded = keras.layers.Dense(1, activation='relu')(decoded)

            #model = keras.models.Model(input, decoded)
            #model.add(MyLayer(10))
            #num_encoder_tokens=1
            latent_dim=1
            #num_decoder_tokens=1

            encoder_inputs = keras.layers.Input(shape=(None,num_encoder_tokens ))
            encoder = keras.layers.LSTM(latent_dim, return_state=True)
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            # We discard `encoder_outputs` and only keep the states.
            encoder_states = [ state_h, state_c ]

            # Set up the decoder, using `encoder_states` as initial state.
            decoder_inputs = keras.layers.Input(shape=(None,num_decoder_tokens))
            # We set up our decoder to return full output sequences,
            # and to return internal states as well. We don't use the
            # return states in the training model, but we will use them in inference.
            decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                                 initial_state=encoder_states)
            decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)

            # Define the model that will turn
            # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
            model = keras.models.Model([ encoder_inputs, decoder_inputs ], decoder_outputs)

            #model = keras.models.Sequential()

            #model.add(keras.layers.GRU(100, input_shape=(dataX.shape[ 1 ], dataX.shape[2])))
            #model.add(keras.layers.Dense(50,activation='tanh', recurrent_activation='hard_sigmoid'))
            #model.add(keras.layers.Dense(10,activation='tanh', recurrent_activation='hard_sigmoid'))
            #model.add(keras.layers.Dense(1,activation='tanh', recurrent_activation='hard_sigmoid'))
            #model.layers[ 0 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])
            #model.layers[ 1 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])
            #model.layers[ 2 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])
            #model.layers[ 3 ].kernel_initializer = SplinesCoef(partitionsX[ 0 ], partitionsY[ 0 ])

            # Compile model
            #model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adagrad())

            model.compile(optimizer=keras.optimizers.Adagrad(), loss='mse')
            model.fit([ encoder_input_data, decoder_input_data ], decoder_target_data,
                         batch_size=64,
                         epochs=60,
                         validation_split=0.3,
                         )


            encoder_model = keras.models.Model(encoder_inputs, encoder_states)
            decoder_state_input_h =keras.layers.Input(shape=(latent_dim,))
            decoder_state_input_c =keras.layers.Input(shape=(latent_dim,))
            decoder_states_inputs = [ decoder_state_input_h, decoder_state_input_c ]
            decoder_outputs, state_h, state_c = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [ state_h, state_c ]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_model = keras.models.Model([ decoder_inputs ] + decoder_states_inputs,
                [ decoder_outputs ] + decoder_states)

            # Reverse-lookup token index to decode sequences back to
            # something readable.
            reverse_input_num_index = dict(
                (i, num) for num, i in input_token_index.items())
            reverse_target_num_index = dict(
                (i, num) for num, i in target_token_index.items())

            return model,encoder_model, decoder_model , reverse_input_num_index,reverse_target_num_index

        seed = 7
        numpy.random.seed(seed)
        #for idx, pCurLbl in enumerate(partition_labels):

            #kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

            #for train, test in kfold.split(partitionsX[idx],partitionsY[idx]):
        #for idx, pCurLbl in enumerate(partition_labels):
        genModel,encoder_model,decoder_model,reverse_input_num_index,reverse_target_num_index =\
        baseline_model()
            #keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, epochs=30)
        self.genModel=genModel
        self.encoder_model=encoder_model
        self.decoder_model=decoder_model
        self.reverse_input_num_index=reverse_input_num_index
        self.reverse_target_num_index=reverse_target_num_index
        self.num_decoder_tokens=num_decoder_tokens
        self.target_token_index=target_token_index

        def decode_sequence(input_seq,max_decoder_seq_length):
            # Encode the input as state vectors.

            states_value =encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            # Populate the first character of target sequence with the start character.
            #target_seq[ 0, 0, target_token_index[ '\t' ] ] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = [ ]
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict(
                    [ target_seq ] + states_value)
                v=genModel.predict([input_seq,output_tokens])
                # Sample a token
                sampled_token_indexv = np.argmax(v[ 0, -1, : ])
                sampled_numv = reverse_target_num_index[ sampled_token_indexv]

                sampled_token_index = np.argmax(output_tokens[ 0, -1, : ])
                sampled_num = reverse_target_num_index[ sampled_token_index ]

                decoded_sentence.append(sampled_num)

                # Exit condition: either hit max length
                # or find stop character.
                if (len(decoded_sentence) > max_decoder_seq_length):
                    stop_condition = True

                # Update the target sequence (of length 1).
                #target_seq = np.zeros((1, 1, num_decoder_tokens))
                #target_seq[ 0, 0, sampled_token_index ] = 1.

                # Update states
                states_value = [ h, c ]

            return decoded_sentence

        #for i in range(0,len(encodedX)):
            #genModel.fit([ np.array(encodedX[i]).reshape(21,encodedX[i].shape[1],1), decodedY_inp[i].reshape(21,decodedY_inp[i].shape[1],1) ], np.array(decodedY_trg[i]).reshape(21,decodedY_trg[i].shape[1],1),
            #batch_size = 1,
            #epochs = 10,
            #validation_split = 0.2)
            #genModel.fit(np.array(dataX),np.array(dataY))
                    #scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
                    #print("%s: %.2f%%" % ("acc: ", scores))


        #for idx, pCurLbl in enumerate(partition_labels):
            #genModel = keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, epochs=1,
                                                                  #verbose=0)
            #partitionsX[ idx ] = partitionsX[ idx ].reshape((partitionsX[ idx ].shape[ 0 ], 1, partitionsX[ idx ].shape[ 1 ]))
            #genModel.fit(np.array(partitionsX[ idx ]), np.array(partitionsY[ idx ]))
            # scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
            # print("%s: %.2f%%" % ("acc: ", scores))

            #models.append(genModel)
        self._partitionsPerModel[ genModel ] = dataX
        models.append(genModel)
        #model=stackedModels(models)
        # Update private models
        #models=[]
        #models.append(model)
        self._models = models
        self._decode_seq=decode_sequence

        # Return list of models
        return models, numpy.empty, numpy.empty , None



    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):

        models = [ ]
        self.ClustersNum=len(partitionsX)
        partition_labels=partitionsX
        # Init model to partition map
        self._partitionsPerModel = {}
        def SplinesCoef(partitionsX, partitionsY):

           model= sp.Earth(use_fast=True)
           model.fit(partitionsX,partitionsY)

           return model.coef_

        class MyLayer(tf.keras.layers.Layer):

            def __init__(self, output_dim):
                self.output_dim = output_dim
                super(MyLayer, self).__init__()

            def build(self, input_shape):

                # Create a trainable weight variable for this layer.
                self.kernel = self.add_weight(name='kernel',
                                              shape=(input_shape[ 1 ], self.output_dim),
                                              trainable=True)
                super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

            def call(self, x,mask=None):
                return keras.backend.dot(x, self.kernel)

            def compute_output_shape(self, input_shape):
                return (input_shape[ 0 ], self.output_dim)

        def stackedModels(members):
            for i,pCurLbl in  enumerate(partition_labels):
                model = members[ i ]
                for layer in model.model.layers:
                    # make not trainable
                    layer.trainable = False
                    # rename to avoid 'unique layer name' issue
                    #layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
                # define multi-headed input
            ensemble_visible = [ model.model.input for model in members ]
            # concatenate merge output from each model
            ensemble_outputs = [ model.model.output for model in members ]
            merge = keras.layers.concatenate(ensemble_outputs)
            hidden = keras.layers.Dense(10, activation='relu')(merge)
            output = keras.layers.Dense(1, activation='relu')(hidden)
            model = keras.models.Model(inputs=ensemble_visible, outputs=output)
            # plot graph of ensemble
            #keras.utils.plot_model(model, show_shapes=True, to_file='/home/dimitris/model_graph.png')
            model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adagrad())
            return model

        def baseline_model():
            # create model
            model = keras.models.Sequential()
            #input_img = keras.layers.Input(shape=(2880,), name='input')
            #x = input_img
            #model.add(keras.layers.Embedding(partitionsX.shape[1], 1, input_length=2,name='layer_0'))
            #model.add(keras.layers.Dense(partitionsX.shape[1],input_dim=1,activation='relu',name='layer_0'))
            #model.add(keras.layers.Dense(partitionsX.shape[1],input_dim=1, activation='relu', name='layer_0'))
            #model.add(keras.layers.Reshape((1,SplineWeights.shape[1],),name='reshape_1'))
            #model.add(keras.layers.LSTM(300,activation='relu'))
            #model.add(keras.layers.Reshape((1, 200,), name='reshape_1'))
            #model.add(keras.layers.Dense(2860,input_dim=1,name='layer_0'))
            #model.add(keras.layers.LSTM(20, activation='relu'))
            #model.add(keras.layers.LSTM(300, activation='relu'))
            # model.add(keras.layers.Dense(pre_trained_weights.shape[0], input_shape=(2,),name='layer_0'))
            model.add(keras.layers.Embedding(partitionsX.shape[ 0 ], 1, input_length=2, name='layer_'))
            model.add(keras.layers.Dense(len(partition_labels), input_shape=(2,), activation='relu', name='layer_0'))
            model.add(keras.layers.Dense(len(partition_labels)*2, activation='relu'))
            model.add(keras.layers.Dense(len(partition_labels)*3, activation='relu'))
            model.add(keras.layers.Dense(len(partition_labels)*4, activation='relu'))
            model.add(keras.layers.Dense(len(partition_labels)*3, activation='relu'))
            model.add(keras.layers.Dense(len(partition_labels)*2, activation='relu'))
            model.add(keras.layers.Dense(len(partition_labels), activation='relu'))
            #model.add(keras.layers.Dropout(0.005))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1))

            # Compile model
            model.compile(loss='mse', optimizer=keras.optimizers.Adam())
            #w_l =[np.array(pre_trained_weights).reshape(1,-1),np.array(pre_trained_weights)]
            #w_l.append(pre_trained_weights)
            #w_l_s = [ np.array(SplineWeights).reshape(1, -1), np.array(SplineWeights).reshape(-1)]
            #w=[np.array(pre_trained_weights),np.array(pre_trained_weights)]
            #w=[np.array(pre_trained_weights)]
            #w=np.array(w).reshape(2860,1)
            #w_l = [ np.array(pre_trained_weights).reshape(1, -1), np.array(pre_trained_weights).reshape(-1) ]
            #w=SplineWeights.reshape(-1,1)
            #w=pre_trained_weights.reshape(-1,1)
                #.reshape(-1,1)
            #model.get_layer(name='layer_0').set_weights([w])
                                                         #np.zeros(partitionsX.shape[1])])
            return model

        def preTrainedWeights():
            from scipy.spatial import Delaunay, ConvexHull
            import math
            from trianglesolver import solve , degree
            dataXnew=partitionsX
            dataYnew=partitionsY
            triNew = Delaunay(dataXnew)

            weights=[]
            indexes=[]
            for k in range(0, len(triNew.vertices)):
                # simplicesIndexes
                #k=triNew.find_simplex(dataXnew[i])
                V1 = dataXnew[ triNew.vertices[ k ] ][ 0 ]
                V2 = dataXnew[ triNew.vertices[ k ] ][ 1 ]
                V3 = dataXnew[ triNew.vertices[ k ] ][ 2 ]

                # plt.plot(candidatePoint[ 0 ], candidatePoint[ 1 ], 'o', markersize=8)
                # plt.show()
                x = 1
                #b = triNew.transform[ k, :2 ].dot(candidatePoint - triNew.transform[ k, 2 ])
                #W1 = b[ 0 ]
                #W2 = b[ 1 ]
                #W3 = 1 - np.sum(b)
                a=spatial.distance.euclidean(V1,V2)
                b = spatial.distance.euclidean(V1, V3)
                c = spatial.distance.euclidean(V3, V2)
                a, b, c, A, B, C = solve(b=a, c=b, a=c)
                A=A/degree
                B=B/degree
                C=C/degree
                W1=math.sin(A)
                W2=math.sin(B)
                W3=math.sin(C)

                if 1==1:

                    rpm1 = dataYnew[ triNew.vertices[ k ] ][ 0 ]
                    rpm2 = dataYnew[ triNew.vertices[ k ] ][ 1 ]
                    rpm3 = dataYnew[ triNew.vertices[ k ] ][ 2 ]


                    x = 0

                    ##barycenters (W12,W23,W13) of neighboring triangles and solutions of linear 3x3 sustem in order to find gammas.


                    #############################
                    ##end of barycenters
                    flgExc = False
                    try:

                        neighboringVertices1 = [ ]


                        for u in range(0, 2):
                            try:
                                neighboringVertices1.append(triNew.vertex_neighbor_vertices[ 1 ][
                                                            triNew.vertex_neighbor_vertices[ 0 ][
                                                                triNew.vertices[ k ][ u ] ]:
                                                            triNew.vertex_neighbor_vertices[ 0 ][
                                                                triNew.vertices[ k ][ u ] + 1 ] ])
                            except:
                                break
                        neighboringTri = triNew.vertices[
                            triNew.find_simplex(dataXnew[ np.concatenate(np.array(neighboringVertices1)) ]) ]
                        if (1==1):

                            nRpms = [ ]
                            nGammas = [ ]

                            rpms = [ ]
                            for s in neighboringTri:
                                V1n = dataXnew[ s ][ 0 ]
                                V2n = dataXnew[ s ][ 1 ]
                                V3n = dataXnew[ s ][ 2 ]

                                rpm1n = dataYnew[ s ][ 0 ]
                                rpm2n = dataYnew[ s ][ 1 ]
                                rpm3n = dataYnew[ s ][ 2 ]

                                rpms.append([ rpm1n, rpm2n, rpm3n ])
                                ###barycentric coords of neighboring points in relation to initial triangle
                                eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                                 [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                                eq2 = np.array([ V1n[ 0 ] - V3[ 0 ], V1n[ 1 ] - V3[ 1 ] ])
                                solutions = np.linalg.solve(eq1, eq2)

                                W1n = solutions[ 0 ]
                                W2n = solutions[ 1 ]
                                W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

                                B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                nRpms.append(rpm1n - B1)

                                nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
                                ####################################

                                eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                                 [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                                eq2 = np.array([ V2n[ 0 ] - V3[ 0 ], V2n[ 1 ] - V3[ 1 ] ])
                                solutions = np.linalg.solve(eq1, eq2)

                                W1n = solutions[ 0 ]
                                W2n = solutions[ 1 ]
                                W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

                                B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                nRpms.append(rpm2n - B1)

                                nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
                                ##################################################

                                eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                                                 [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

                                eq2 = np.array([ V3n[ 0 ] - V3[ 0 ], V3n[ 1 ] - V3[ 1 ] ])
                                solutions = np.linalg.solve(eq1, eq2)

                                W1n = solutions[ 0 ]
                                W2n = solutions[ 1 ]
                                W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

                                B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
                                nRpms.append(rpm3n - B1)
                                nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))


                            nGammas = np.array(nGammas)
                            nRpms = np.array(nRpms)
                            from sklearn.linear_model import LinearRegression
                            lr = LinearRegression()
                            leastSqApprx = lr.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
                            XpredN = ((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) +
                                      2 * W1 * W2 * leastSqApprx.coef_[ 0 ][ 0 ] +
                                      2 * W2 * W3 * leastSqApprx.coef_[ 0 ][ 1 ] +
                                      2 * W1 * W3 * leastSqApprx.coef_[ 0 ][ 2 ])
                            weights.append(leastSqApprx.coef_[0][0])
                            weights.append(leastSqApprx.coef_[ 0 ][ 1 ])
                            weights.append(leastSqApprx.coef_[ 0 ][ 2 ])
                            #weights.append(np.mean(leastSqApprx.coef_))

                    except :
                        #print(str(e))
                        x=1
            return weights
        seed = 7
        numpy.random.seed(seed)
        #for idx, pCurLbl in enumerate(partition_labels):

            #kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
        #genModel = baseline_model()
        partitionsX=X
        partitionsY=Y
        #weights = preTrainedWeights()
        #weights=np.array(weights)[0:partitionsX.shape[0]]

        #pre_trained_weights = self.createModelsForAUTO_ENC(partitionsX, partitionsY, partition_labels)
        #pre_trained_weights=weights
        #CLpartitionsX, CLpartitionsY, partition_labels  = self.createModelsForAUTO(partitionsX, partitionsY,
                                                                              #partition_labels)


        #estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=baseline_model, epochs=100,)
        estimator = baseline_model()
        #partitionsX=partitionsX.reshape(-1,2)
        #partitionsY = partitionsY.reshape(-1, 1)

        #X_train, X_test, y_train, y_test = train_test_split(partitionsX, partitionsY, test_size=0.33, random_state=seed)
        #partitionsX=np.reshape(partitionsX, (partitionsX.shape[ 0 ], 1, partitionsX.shape[ 1 ]))
        estimator.fit(partitionsX,partitionsY,epochs=100,validation_split=0.33,batch_size=100)




            #for train, test in kfold.split(partitionsX[idx],partitionsY[idx]):
        #if len(partition_labels)>1:
            #for idx, pCurLbl in enumerate(partition_labels):
                #partitionsX[ idx ]=partitionsX[idx].reshape(-1,2)
                #estimator.fit(np.array(CLpartitionsX[idx]),np.array(CLpartitionsY[idx]),validation_split=0.33,epochs=100,batch_size=100)
                    #scores = estimator.score(partitionsX[idx][ test ], partitionsY[idx][ test ])
                    #print("%s: %.2f%%" % ("acc: ", scores))


            #self._partitionsPerModel[ estimator ] = partitionsX[idx]
        #model=stackedModels(models)
        # Update private models
        #models=[]
        #models.append(model)
        models.append(estimator)
        self._models = models

        # Return list of models
        return estimator, numpy.empty, numpy.empty , None

    def createModelsForConv(self,partitionsX, partitionsY, partition_labels):
        ##Conv1D NN
        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}

        TIME_PERIODS=len(partitionsX[0])
        STEP_DISTANCE = 40
        x_train, y_train = self.createSegmentsofTrData(partitionsX[ 0 ], partitionsY[ 0 ], TIME_PERIODS, STEP_DISTANCE)

        num_time_periods, num_sensors =partitionsX[0].shape[0], partitionsX[0].shape[1]
        num_classes = 1
        input_shape = (num_sensors)
        #x_train = x_train.reshape(x_train.shape[ 0 ], input_shape)


        model_m = keras.models.Sequential()
        model_m.add(keras.layers.Reshape((TIME_PERIODS,num_sensors, ), input_shape=(num_sensors,)))
        model_m.add(keras.layers.Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
        model_m.add(keras.layers.Conv1D(100, 10, activation='relu'))
        model_m.add(keras.layers.MaxPooling1D(2))
        model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
        model_m.add(keras.layers.Conv1D(160, 10, activation='relu'))
        model_m.add(keras.layers.GlobalAveragePooling1D())
        model_m.add(keras.layers.Dropout(0.3))
        model_m.add(keras.layers.Dense(num_classes, activation='softmax'))
        print(model_m.summary())

        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]

        model_m.compile(loss='mean_squared_error',
                        optimizer='adam', metrics=[ 'accuracy' ])

        BATCH_SIZE = 400
        EPOCHS = 50
        for idx, pCurLbl in enumerate(partition_labels):
           #for i in range(0,x_train.shape[1]):
           history = model_m.fit(np.nan_to_num(partitionsX[idx]),np.nan_to_num(partitionsY[idx]),
                                      epochs=EPOCHS,
                                     )

        models.append(model_m)
        self._partitionsPerModel[ model_m ] = partitionsX[ idx ]

        # Update private models

        self._models = models

        # Return list of models
        return models, numpy.empty, numpy.empty


    def createModelsForS(self,partitionsX, partitionsY, partition_labels):

        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        #curModel = keras.models.Sequential()

        # This will add a fully connected neural network layer with 32#neurons, each#taking#13#inputs, and with activation function ReLU
        #curModel.add(keras.layers.Dense(1, input_dim=3, activation='relu'))

        #curModel.compile(loss='mean_squared_error',
        #                 optimizer='sgd',
        #                 metrics=[ 'mae' ])
        # Fit to data

        #input_A = keras.layers.Input(shape=partitionsX[0].shape)
        #input_B = keras.layers.Input(shape=partitionsX[1].shape)
        #input_C = keras.layers.Input(shape=partitionsX[2].shape)
        #A = keras.layers.Dense(1)(input_A)
        #A = keras.layers.Dense(1)(A)
        #B = keras.layers.concatenate([ input_A, input_B, input_C ], mode='concat')
        #B = keras.layers.Dense(1)(B)
        #C = keras.layers.concatenate([ A, B ], mode='concat')
        #C = keras.layers.Dense(1)(C)
        #B = keras.layers.concatenate([ A, B ], mode='concat')
        #B = keras.layers.Dense(1)(B)
        #A = keras.layers.Dense(1)(A)

        #curModel.fit(np.asarray([ [[0,2], [0,2] ],[[0,2], [0,2] ], [[0,2], [0,2] ] ]),
                     #np.asarray([ [[0,2], [0,2] ],[[0,2], [0,2] ], [[0,2], [0,2] ] ]), epochs=10)
        #curModel.fit(partitionsX, partitionsY, epochs=10)

        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = keras.models.Sequential()
            partitionsX[ idx ] = np.reshape(partitionsX[ idx ], (partitionsX[ idx ].shape[ 0 ], 1, partitionsX[ idx ].shape[ 1 ]))

            # This will add a fully connected neural network layer with 32#neurons, each#taking#13#inputs, and with activation function ReLU
            curModel.add(keras.layers.Conv1D(2,input_shape=partitionsX[idx].shape[1:],activation='relu'))
            #curModel.add(keras.layers.LSTM(3, input_shape=partitionsX[ idx ].shape[ 1: ], activation='relu'))
            #curModel.add(keras.layers.Flatten())
            curModel.add(keras.layers.Dense(1))

            ##optimizers
            adam=keras.optimizers.Adam(lr=0.001)
            rmsprop=keras.optimizers.RMSprop(lr=0.01)
            adagrad=keras.optimizers.Adagrad(lr=0.01)
            sgd=keras.optimizers.SGD(lr=0.001)
            ######

            curModel.compile(loss='mean_squared_error',
              optimizer=adam,metrics=['mae'])
            # Fit to data
            curModel.fit(np.nan_to_num(partitionsX[idx]),np.nan_to_num(partitionsY[idx]), epochs=100)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[curModel] = partitionsX[idx]

        # Update private models
        self._models = models

        # Return list of models
        return models , numpy.empty ,numpy.empty


    def createModelsForX1(self, partitionsX, partitionsY, partition_labels):

        SAVE_PATH = './save'
        EPOCHS = 1
        LEARNING_RATE = 0.001
        MODEL_NAME = 'test'

        model = sp.Earth(use_fast=True)
        model.fit(partitionsX[0],partitionsY[0])
        W_1= model.coef_

        self._partitionsPerModel = {}


        ######################################################
        # Data specific constants
        n_input = 784  # data input 
        n_classes = 1  #  total classes
        # Hyperparameters
        max_epochs = 10
        learning_rate = 0.5
        batch_size = 10
        seed = 0
        n_hidden = 3  # Number of neurons in the hidden layer


        # Gradient Descent optimization  for updating weights and biases
        # Execution Graph
        c_t=[]
        c_test=[]
        models=[]

        for idx, pCurLbl in enumerate(partition_labels):
            n_input = partitionsX[ idx ].shape[ 1 ]
            xs = tf.placeholder(tf.float32, [ None, n_input ])
            ys = tf.placeholder("float")

            output = self.initNN(xs, 2)
            cost = tf.reduce_mean(tf.square(output - ys))
            # our mean square error cost function
            train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
            ##

            init_op = tf.global_variables_initializer()
            #name_to_var_map = {var.op.name: var for var in tf.global_variables()}

            #name_to_var_map[ "loss/total_loss/avg" ] = name_to_var_map[
            #    "conv_classifier/loss/total_loss/avg" ]
            #del name_to_var_map[ "conv_classifier/loss/total_loss/avg" ]

            saver = tf.train.Saver()
            if idx==1:
                with tf.Session() as sess:
                    # Initiate session and initialize all vaiables
                        #################################################################
                        # output = self.initNN(xs,W_1,2)
                        sess.run(init_op)

                        for i in range(EPOCHS):
                      #for j in range(len(partitionsX[idx].shape[ 0 ]):
                          for j in range(0,len(partitionsX[idx])):

                             all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                             tf.variables_initializer(all_variables)
                             sess.run([ cost, train ], feed_dict={xs: [partitionsX[idx][ j, : ]], ys: [partitionsY[idx][ j ]]})
                        # Run cost and train with each sample
                        #c_t.append(sess.run(cost, feed_dict={xs: partitionsX[idx], ys: partitionsY[idx]}))
                        #c_test.append(sess.run(cost, feed_dict={xs: X_test, ys: y_test}))
                        #print('Epoch :', i, 'Cost :', c_t[ i ])
                        models.append(sess)
                        #self._partitionsPerModel[sess]  = partitionsX[idx]

                        self._models = models
                        saver.save(sess, SAVE_PATH + '/' + MODEL_NAME+'_'+str(idx) + '.ckpt')
                sess.close()
        return  models , xs , output

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

class RandomForestModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = skl.RandomForestRegressor()
            ##HP tuning
            #random_search3 = RandomizedSearchCV(curModel, param_distributions=parameters.param_mars_dist, n_iter=4)
            # try:
            #    random_search3.fit(partitionsX[ idx ], partitionsY[ idx ])
            #    curModel.set_params(**self.report(random_search3.cv_results_))
            # except:
            #    print "Error on HP tuning"
            # Fit to data
            try:
                curModel.fit(partitionsX[ idx ], partitionsY[ idx ])
            except:
                print (idx)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models, numpy.empty, numpy.empty , None

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))


class SplineRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels,tri,X,Y):
        # Init result model list
        models = [ ]
        #import scipy as sp1
        dataX = X
        dataY = Y
        #spline = sp1.interpolate.Rbf(x[0:,0], x[0:,1], y,
                                     #function='thin_plate', smooth=5, episilon=5)
        #genericModel = sp.Earth(use_fast=True)
        #genericModel.fit(x,y)


        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            #from sklearn.decomposition import PCA
            #pcaMapping = PCA(1)
            #pcaMapping.fit(partitionsX[ idx ])
            #Vmapped = pcaMapping.transform(partitionsX[ idx ])
            #dataX = Vmapped
            #if(st.pearsonr(np.asarray(dataX).reshape(1,-1)[0], partitionsY[ idx ])[0]>0.5):
                #n=1
            #else:
                #n=2
            curModel = sp.Earth()
            ##HP tuning
            #random_search3 = RandomizedSearchCV(curModel, param_distributions=parameters.param_mars_dist, n_iter=4)
            #try:
            #    random_search3.fit(partitionsX[ idx ], partitionsY[ idx ])
            #    curModel.set_params(**self.report(random_search3.cv_results_))
            #except:
            #    print "Error on HP tuning"
            # Fit to data
            #try:
            #simplicesIndexes = tri.find_simplex(partitionsX[ idx ])
            #triVerticesIndexesFiltered = [x for x in tri.vertices[simplicesIndexes] if int(x) < int(len(dataX))]
            #dataXnew=np.concatenate(dataX[tri.vertices[simplicesIndexes]])
            #dataYnew =np.concatenate(dataY[tri.vertices[simplicesIndexes]])
            curModel.fit(partitionsX[idx],partitionsY[idx])
            #plt.plot(partitionsX[ idx ], partitionsY[ idx ], 'o', markersize=3)
            #plt.plot(partitionsX[ idx ], curModel.predict(partitionsX[ idx ]), 'r--' )

            #plt.show()
            #x=1
            #except:
                #print str(Exception)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models,numpy.empty ,numpy.empty,None

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))

    def plotRegressionLine(self,partitionsX, partitionsY, partitionLabels,genericModel,modelMap):
        #xData = np.asarray(partitionsX[ :, 0 ]).T[ 0 ]
        #yData = np.asarray(partitionsY[ :, 1 ]).T[ 0 ]
        xData = np.concatenate(partitionsX)
        yData = np.concatenate(partitionsY)

        #plt.scatter(xData[:,0], xData[:,1], c=partitionLabels)

        for idx, pCurLbl in enumerate(partitionLabels):
            plt.plot(partitionsX[ idx ][:,0], partitionsY[ idx ], 'o', markersize=3)
            plt.plot(partitionsX[ idx ], modelMap[idx].predict(partitionsX[ idx ]), 'r--')

        plt.plot(xData, genericModel.predict(xData), 'b-')
        plt.show()
        x=1
        return


class CSplineRegressionModeler(BasePartitionModeler):


    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        # Init result model list
        models = []
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):

            mars = sp.Earth()
            mars.fit(partitionsX[idx],partitionsY[idx])
            bvSpl=BivariateSpline()
            curModel= bvSpl.__call__(partitionsX[idx],partitionsY[idx],2,0,True,)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))


class XSplineRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model
            curModel = GEKKO()  # create GEKKO model
            curModel.options.IMODE = 2  # solution mode
            x = curModel.Param(value=partitionsX[idx])  # prediction points
            y = curModel.Var()  # prediction results

            # Fit to data
            curM=GEKKO()
            curM.options.IMODE=2
            x1 = curM.Param(value=np.array(partitionsX[ idx ][:,0]).reshape(1,-1)[0])  # prediction points
            y1 = curM.Var()  # prediction results
            curM.cspline(x1, y1, np.array(partitionsX[ idx ][:,0]).reshape(1,-1)[0], np.array(partitionsX[ idx ][:,0]).reshape(1,-1)[0])  # cubic spline
            curM.solve(disp=False)

            curModel.cspline(x, y,y1, partitionsY[ idx ])  # cubic spline
            curModel.solve(disp=False)
            # Add to returned list
            plt.plot(partitionsX[ idx ], partitionsY[ idx ], 'bo')
            plt.plot(x, y.value, 'r--', label='cubic spline')
            plt.legend(loc='best')
            plt.show()

            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models

    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[ model ]) - point))



class NCSRegressionModeler(BasePartitionModeler):
    def createModelsFor(self, partitionsX, partitionsY, partition_labels):
        knots=12

        def make_natural_cubic_regression(n_knots):
                return Pipeline([
                    ('standardizer', StandardScaler()),
                    ('nat_cubic', NaturalCubicSpline(-2, 2, n_knots=n_knots)),
                    ('regression', LinearRegression(fit_intercept=True))
                ])

        # Init result model list
        models = [ ]
        # Init model to partition map
        self._partitionsPerModel = {}
        # For each label/partition
        for idx, pCurLbl in enumerate(partition_labels):
            # Create a linear model

            curModel = Pipeline([
                    ('standardizer', StandardScaler()),
                    ('nat_cubic', sp.Earth(max_degree=3, penalty=11.0,smooth=True,allow_linear=False)),
                    ('regression', LinearRegression(fit_intercept=True))
                ])
            ##HP tuning
            #random_search3 = RandomizedSearchCV(curModel, param_distributions=parameters.param_mars_dist, n_iter=4)
            # try:
            #    random_search3.fit(partitionsX[ idx ], partitionsY[ idx ])
            #    curModel.set_params(**self.report(random_search3.cv_results_))
            # except:
            #    print "Error on HP tuning"
            # Fit to data
            try:
                curModel.fit(partitionsX[ idx ], partitionsY[ idx ])
            except:
                print (idx)
            # Add to returned list
            models.append(curModel)
            self._partitionsPerModel[ curModel ] = partitionsX[ idx ]

        # Update private models
        self._models = models

        # Return list of models
        return models



    def getFitnessOfModelForPoint(self, model, point):
        return 1.0 / (1.0 + numpy.linalg.norm(np.mean(self._partitionsPerModel[model])-point))

    def report(self,results, n_top=1):
        for i in range(1, n_top + 1):
            candidate = np.flatnonzero(results[ 'rank_test_score' ] == i)
            # print("Mean validation score: {0:.3f} )".format(
            # results[ 'mean_test_score' ][ candidate[0]]))
            # print("Parameters: {0}".format(results[ 'params' ][ candidate[0] ]))
            return results[ 'params' ][ candidate[ 0 ] ]

class MarkovChainModeling(SplineRegressionModeler,LinearRegressionModeler):
    def FindMostrewardingNextTrSet(self,partitionsX):
        pass

    def Initialize(self,partitionsX,partition_labels):
        # The statespace
        states=[]
        for idx, pCurLbl in enumerate(partition_labels):
            states.append( [partitionsX[idx],pCurLbl] )

        # Possible sequences of eventsfor L in range(0, len(stuff)+1):
        for l in range(0, len(states) + 1):
            for subset in itertools.combinations(states, l):
                pass
                #transinions.append()

        # Probabilities matrix (transition matrix)
        transitionMatrix = [ [ 0.2, 0.6, 0.2 ], [ 0.1, 0.6, 0.3 ], [ 0.2, 0.7, 0.1 ] ]

    def FindMostrewardingNextTrSet(self,transMatrx,currPartitionX,partitionsX):
            pass