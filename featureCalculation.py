import numpy as np
from sklearn.decomposition import PCA

class BaseFeatureExtractor:
    # Get a series of tuples, where the i-th tuple contains the current sample value X_i and the average X values over the previous 30 samples before i
    def extractFeatures(self, modeler,X, Y,W,B,history):
        HISTORY_SIZE = 15

        Xnew = []
        Yt=[]
        XnewX=[]
        prev = []
        Y_decInput=[]
        Y_decTrget=[]
        #(B[i-HISTORY_SIZE]-B[i])
        W=None
        for i in range(HISTORY_SIZE,len(X)):
          '''if W is not None:
            if W[i]>22 and W[i] < 27: ## 6 Beaufort
                HISTORY_SIZE=history+5
            elif W[i]>28 and W[i] < 33: ## 7 Beaufort
                HISTORY_SIZE =history+ 10
            elif W[ i ] > 34 and W[ i ] < 40:  ## 8 Beaufort
                HISTORY_SIZE =history+ 20
            elif W[ i ] > 41 and W[ i ] < 47:  ## 9 Beaufort
                HISTORY_SIZE =history+ 30
            elif W[ i ] > 48 and W[ i ] < 55:  ## 10 Beaufort
                HISTORY_SIZE =history+ 40
            elif W[ i ] > 56 and W[ i ] < 63:  ## 11 Beaufort
                HISTORY_SIZE =history+ 50'''
            #*np.mean(W[i-HISTORY_SIZE:i])
            #pcaMapping = PCA(1)
            #pcaMapping.fit(np.array(X[ i - HISTORY_SIZE:i ]).reshape(-1, X[ i - HISTORY_SIZE:i ].size))
            #Vmapped = pcaMapping.transform(np.array(X[ i - HISTORY_SIZE:i ]).reshape(-1, X[ i - HISTORY_SIZE:i ].size))
            #dataX = Vmapped
          #if modeler.__class__.__name__ != 'TensorFlow':
          #prev =np.append([X[i],np.mean(X[ i - HISTORY_SIZE:i ])],np.mean(X[ i - 2*HISTORY_SIZE:i - HISTORY_SIZE ]))
          #else:
          #if X[i] > 1:
          if modeler.__class__.__name__ == 'TensorFlowWLSTM2':
            prev = np.append(X[i],X[ i - HISTORY_SIZE:i ])
            Yt.append(Y[i])
          else:
            prev = np.append(X[i], np.mean(X[i - HISTORY_SIZE:i]))
            Yt.append(Y[i])

          #if modeler.__class__.__name__ == 'TensorFlow':
              #prevY = np.append(Y[ i ], Y[ i - HISTORY_SIZE:i ])
              #if i+1 <  len(X):
                  #prevYt = np.append(Y[ i+1 ], Y[ (i+1) - HISTORY_SIZE:(i+1) ])
            # for i in range(HISTORY_SIZE, len(X))
          Xnew.append(prev)
          #Y_decInput.append(prevY)
          #Y_decTrget.append(prevYt)
        if modeler.__class__.__name__ != 'TensorFlowWLSTM2':
            Xnew = np.array(Xnew).reshape(-1, 2)
        else:
            Xnew = np.array(Xnew).reshape(-1, 16)
        #Xnew = np.array(X).reshape(-1, 1)

        #Wnew = [ ]
        #if self.__class__.__name__=="BaseFeatureExtractor":
        #    prev = [ [ W[ i ], np.mean(W[ i - HISTORY_SIZE:i ]) ] for i in range(HISTORY_SIZE, len(W)) ]
        #    Wnew.append(prev)
        #    Wnew = np.array(Wnew).reshape(-1, 2)

        # Return data (ignoring lines that have no target value)

        #if modeler.__class__.__name__ != 'TensorFlow':
        #Y[ history: ]
        return Xnew, np.array(Yt), []
        #else:
            #return Xnew, Y_decInput, Y_decTrget

    def extractFeaturesForLaros(self, modeler,X, Y,W,B,history):
        HISTORY_SIZE = 35

        Xnew = []
        Yt=[]
        XnewX=[]
        prev = []
        Y_decInput=[]
        Y_decTrget=[]
        #(B[i-HISTORY_SIZE]-B[i])
        W=None
        for i in range(HISTORY_SIZE,len(X)):
          if W is not None:
            if W[i]>22 and W[i] < 27: ## 6 Beaufort
                HISTORY_SIZE=history+5
            elif W[i]>28 and W[i] < 33: ## 7 Beaufort
                HISTORY_SIZE =history+ 10
            elif W[ i ] > 34 and W[ i ] < 40:  ## 8 Beaufort
                HISTORY_SIZE =history+ 20
            elif W[ i ] > 41 and W[ i ] < 47:  ## 9 Beaufort
                HISTORY_SIZE =history+ 30
            elif W[ i ] > 48 and W[ i ] < 55:  ## 10 Beaufort
                HISTORY_SIZE =history+ 40
            elif W[ i ] > 56 and W[ i ] < 63:  ## 11 Beaufort
                HISTORY_SIZE =history+ 50
            #*np.mean(W[i-HISTORY_SIZE:i])
            #pcaMapping = PCA(1)
            #pcaMapping.fit(np.array(X[ i - HISTORY_SIZE:i ]).reshape(-1, X[ i - HISTORY_SIZE:i ].size))
            #Vmapped = pcaMapping.transform(np.array(X[ i - HISTORY_SIZE:i ]).reshape(-1, X[ i - HISTORY_SIZE:i ].size))
            #dataX = Vmapped
          #if modeler.__class__.__name__ != 'TensorFlow':
          #prev =np.append([X[i],np.mean(X[ i - HISTORY_SIZE:i ])],np.mean(X[ i - 2*HISTORY_SIZE:i - HISTORY_SIZE ]))
          #else:
          if X[i] > 1:
            prev = np.append(X[i],np.mean(X[ i - HISTORY_SIZE:i ]))
            Yt.append(Y[i])
          #if modeler.__class__.__name__ == 'TensorFlow':
              #prevY = np.append(Y[ i ], Y[ i - HISTORY_SIZE:i ])
              #if i+1 <  len(X):
                  #prevYt = np.append(Y[ i+1 ], Y[ (i+1) - HISTORY_SIZE:(i+1) ])
            # for i in range(HISTORY_SIZE, len(X))
            Xnew.append(prev)
          #Y_decInput.append(prevY)
          #Y_decTrget.append(prevYt)
        #if modeler.__class__.__name__ != 'TensorFlow':
        Xnew = np.array(Xnew).reshape(-1, 2)
        #Xnew = np.array(X).reshape(-1, 1)

        #Wnew = [ ]
        #if self.__class__.__name__=="BaseFeatureExtractor":
        #    prev = [ [ W[ i ], np.mean(W[ i - HISTORY_SIZE:i ]) ] for i in range(HISTORY_SIZE, len(W)) ]
        #    Wnew.append(prev)
        #    Wnew = np.array(Wnew).reshape(-1, 2)

        # Return data (ignoring lines that have no target value)

        #if modeler.__class__.__name__ != 'TensorFlow':
        #Y[ history: ]
        return Xnew, np.array(Yt), []
        #else:
            #return Xnew, Y_decInput, Y_decTrget

    def extractFeatureswithVariance(self,X,Y,futureN):
        FUTUREv = futureN
        Xnew = [ ]
        next = [ [ X[ i ], np.var(X[ i:i+FUTUREv ]) ] for i in range(0, len(X)) ]
        Xnew.append(next)
        Xnew = np.array(Xnew).reshape(-1, 2)

        # Return data (ignoring lines that have no target value)
        return Xnew,Y

class UnseenFeaturesExtractor(BaseFeatureExtractor):

        pass