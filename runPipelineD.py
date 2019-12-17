import numpy as np
import sys
import pandas as pd
from tensorflow import keras
from scipy.spatial import distance
import argparse
from sklearn.externals import joblib

def main():
    # Init parameters based on command line
    listOfPoints , path , clusters = initParameters()
    # Load data
    clusters = int(clusters)
    #print(path)
    partitionsX=[]
    partitionsY=[]
    preds = [ ]
    listOfPoints = np.array(listOfPoints.split('[')[ 1 ].split(']')[ 0 ].split(',')).astype(np.float)
    listOfPoints = listOfPoints.reshape(-1, 5)
    for cl in range(0,50):
        data = pd.read_csv('cluster_'+str(cl)+'_.csv')
        partitionsX.append(readClusteredLarosDataFromCsvNew(data))

    #for cl in range(0,50):
        #data = pd.read_csv('cluster_foc'+str(cl)+'_.csv')
        #partitionsY.append(readClusteredLarosDataFromCsvNew(data))

    data = pd.read_csv('dataX_.csv')
    X = (readClusteredLarosDataFromCsvNew(data))

    data = pd.read_csv('dataY_.csv')
    Y = (readClusteredLarosDataFromCsvNew(data))

    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scalerY = StandardScaler()
    scalerX = scalerX.fit(X)
    scalerY = scalerY.fit(Y.reshape(-1, 1))

    scaler_x_filename = args.Scaler_X
    Scaler_x = joblib.load(scaler_x_filename)

    scaler_y_filename = args.Scaler_Y
    Scaler_y = joblib.load(scaler_y_filename)

    for vector in listOfPoints:
        #ind, fit =getBestPartitionForPoint(vector, partitionsX)
        #currModeler = keras.models.load_model('estimatorCl_rpm' + str(ind) + '.h5')
        vector = vector.reshape(-1, 5)
        #rpm = currModeler.predict(vector)
        # preds.append(prediction[ 0 ][ 0 ])
        #pPoint = np.append(listOfPoints, [ rpm ])

    #print('\nVector/list of vectors for prediction: ' + str(listOfPoints))
    #for vector in listOfPoints:
        #vector = pPoint
        scaled = scalerX.fit_transform(vector)

        #vctr = np.array([ vector[ 0 ][ 0 ], vector[ 0 ][ 4 ] ]).reshape(-1, 2)
        ind, fit =  getBestPartitionForPoint(vector, partitionsX)

        scaled = Scaler_x.fit_transform(vector)

        currModeler = keras.models.load_model( 'estimatorCl_' + str(ind) + '.h5')
        currModeler1 = keras.models.load_model('estimatorCl_Gen_.h5')
        vector = vector.reshape(-1, 5)
        prediction = currModeler.predict(vector)

        prediction1 = currModeler1.predict(vector)

        #prediction = (Scaler_y.inverse_transform(prediction))  # + scalerY.inverse_transform(currModeler1.predict(scaled))) / 2
        #prediction1 = (Scaler_y.inverse_transform(prediction1))  # + scalerY.inverse_transform(currModeler1.predict(scaled))) / 2
        if vector[0][0] < 6:
            scaledPred=(prediction + prediction*fit) - 5
        else:
            scaledPred=(prediction + prediction*fit) + 5

        preds.append(prediction[0][0])

    print('\n'+'FOC (tons / day) : '+str(np.round(scaledPred * 1.44,3)))
    return preds

def initParameters():
    sFile = "./kaklis.csv"
    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000
    algs=['NNWD']
    cls=['KM']

    if len(sys.argv) > 1:
        x=0
        #listOfPoints = sys.argv[1]
        #path = sys.argv[2]
        #clusters = int(sys.argv[3])
        #history = sys.argv[2]
        #start = sys.argv[3]
        #end = sys.argv[4]
        #algs=sys.argv[5]
        #cls=sys.argv[6]
        listOfPoints = "[3.3956,-3,2.2727,-9.7,9]"
        clusters = '50'
        path = '.\\'
    else:
        listOfPoints = "[7.3956,-3,8.2727,-9.7,9]"
        path = '.\\'
        clusters = '50'

    return listOfPoints, path , clusters

def getFitForEachPartitionForPoint( point, partitions):
        # For each model
        fits=[]
        for m in range(0, len(partitions)):
            # If it is a better for the point
            dCurFit = getFitnessOfPoint(partitions, m, point)
            fits.append(dCurFit)


        return fits

def getFitnessOfPoint(partitions, cluster, point):
    #return distance.euclidean(np.mean(partitions[cluster], axis=0) , point)
    #return 1 / (1 +  np.linalg.norm(np.mean(np.array(np.append(partitions[ cluster ][ :,0 ].reshape(-1, 1), np.asmatrix(partitions[ cluster ][ :, 4 ]).T, axis=1)),axis=0)- np.array([point[0][0],point[0][4]])))

    return 1 / (1 + np.linalg.norm(np.mean(np.array(partitions[ cluster ][ :,4 ].reshape(-1, 1))) - np.array(point[ 0 ][ 4 ] )))
    #return 1 / (1 + np.linalg.norm(np.mean(np.array(
        #np.append(partitions[ cluster ][ :, 0 ].reshape(-1, 1),
                  #np.asmatrix([partitions[ cluster ][ :, 2 ], partitions[ cluster ][ :, 4 ]]).T,
                  #axis=1)),axis=0) - np.array([ point[ 0 ][ 0 ], point[ 0 ][ 2 ], point[ 0 ][ 4 ] ])))
       #return 1 / (1 + np.linalg.norm(np.mean(partitions[cluster], axis=0) - point))
       #return 1 / (1 + np.linalg.norm(np.mean(partitions[cluster][:,1],axis=0) -point[:,1]))

def getBestPartitionForPoint(point,partitions):
            # mBest = None
            mBest = None
            dBestFit = 0
            # For each model
            for m in range(0,len(partitions)):
                # If it is a better for the point
                dCurFit = getFitnessOfPoint(partitions,m, point)
                if dCurFit > dBestFit:
                    # Update the selected best model and corresponding fit
                    dBestFit = dCurFit
                    mBest = m

            if mBest == None:
                return 0,0
            else:
                return mBest , dBestFit

def readClusteredLarosDataFromCsvNew( data):
        # Load file
        dtNew = data.values[0:, 0:5].astype(float)
        dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]
        seriesX = dtNew
        UnseenSeriesX = dtNew
        return UnseenSeriesX

def readLarosDataFromCsvNew(data):
        # Load file

        dataNew=data.drop(['M/E FOC (kg/min)' ,'DateTime'],axis=1)
        dtNew = dataNew.values[ 0:, 0:6 ].astype(float)
        dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]
        seriesX=dtNew[0:140000,:]
        UnseenSeriesX = dtNew[150000:151000, :]

        dt = data.values[0:, 5].astype(float)
        dt = dt[~np.isnan(dt).any(axis=0)]
        FOC = dt[0][0:140000]

        unseenFOC = dt[0][150000:151000]

        return seriesX ,FOC , UnseenSeriesX , unseenFOC

# # ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("Scaler_X", help="Scaler X")
    parser.add_argument("Scaler_Y", help="Scaler Y")

    args = parser.parse_args()
    main()
