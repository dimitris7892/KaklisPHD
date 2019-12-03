import numpy as np
import sys
import pandas as pd
from tensorflow import keras

def main():
    # Init parameters based on command line
    listOfPoints , path , clusters = initParameters()
    # Load data
    clusters = int(clusters)
    #print(path)
    partitionsX=[]
    for cl in range(0,clusters):
        #models.append(load_model(path+'\estimatorCl_'+str(cl)+'.h5'))
        data = pd.read_csv(path+'\cluster_'+str(cl)+'_.csv')
        partitionsX.append(readClusteredLarosDataFromCsvNew(data))

    preds=[]
    listOfPoints = np.array(listOfPoints.split('[')[1].split(']')[0].split(',')).astype(np.float)
    listOfPoints = listOfPoints.reshape(-1,6)
    #print('\nVector/list of vectors for prediction: ' + str(listOfPoints))
    for vector in listOfPoints:
        ind, fit =  getBestPartitionForPoint(vector, partitionsX)
        currModeler = keras.models.load_model(path + '\estimatorCl_' + str(ind) + '.h5')
        vector = vector.reshape(-1, 6)
        prediction = currModeler.predict(vector)
        preds.append(prediction[0][0])

    print('\n'+'FOC (tones / hour) : '+str(preds[0] * 1.44))
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
        listOfPoints = sys.argv[1]
        path = sys.argv[2]
        clusters = int(sys.argv[3])
        #history = sys.argv[2]
        #start = sys.argv[3]
        #end = sys.argv[4]
        #algs=sys.argv[5]
        #cls=sys.argv[6]
    else:
        listOfPoints = "[4.3956,13.2727,-9.7,10.6,582.0,6291.0]"
        path = '.\\'
        clusters = '15'

    return listOfPoints, path , clusters

def getFitForEachPartitionForPoint( point, partitions):
        # For each model
        fits=[]
        for m in range(0, len(partitions)):
            # If it is a better for the point
            dCurFit = getFitnessOfPoint(partitions, m, point)
            fits.append(dCurFit)


        return fits

def getFitnessOfPoint(partitions ,cluster, point):
    return 1.0 / (1.0 + np.linalg.norm(np.mean(partitions[ cluster ]) - point))

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
    main()
