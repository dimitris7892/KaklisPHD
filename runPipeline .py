import dataReading as dRead
import featureCalculation as fCalc
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import numpy as np
from math import sqrt
import sys
import plotResults as plres
import itertools

def main():
    # Init parameters based on command line
    end, endU, history, future,sFile, start, startU = initParameters()
    # Load data
    #KtrSetlen=[[np.square(i) for i  in range(1,8)],[i for i in range(10000,80000,10000)]]
    K=[np.square(i) for i  in range(1,8)]
    trSetlen=[i for i in range(10000,80000,10000)]
    errors=[]
    clusters=[]
    trSize=[]
    #KtrSetlen=zip(K,trSetlen)
    KtrSetlen=list(itertools.product(K, trSetlen))


    for k,trLen in zip(K,trSetlen):
        print("Reading data...")
        reader = dRead.BaseSeriesReader()

        seriesX, targetY = reader.readSeriesDataFromFile(sFile,start,end,trLen)
        print("Reading data... Done.")

        # Extract features
        print("Extracting features from training set...")
        featureExtractor = fCalc.BaseFeatureExtractor()
        X, Y = featureExtractor.extractFeatures(seriesX, targetY,history)
        print("Extracting features from training set... Done.")

        # Partition data
        print("Partitioning training set...")
        NUM_OF_CLUSTERS =k# TODO: Read from command line
        #partitioner = dPart.DefaultPartitioner()
        partitioner = dPart.KMeansPartitioner()
        #partitioner = dPart.BoundedProximityPartitioner()
        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel = partitioner.clustering(X, Y, NUM_OF_CLUSTERS, False)
        # Keep label to partition mapping in a dict
        partitionXPerLabel = dict(zip(partitionLabels, partitionsX))
        partitionYPerLabel = dict(zip(partitionLabels, partitionsY))
        print("Partitioning training set... Done.")

        # For each partition create model
        print("Creating models per partition...")
        modeler = dModel.LinearRegressionModeler()
        # ...and keep them in a dict, connecting label to model
        modelMap = dict(zip(partitionLabels, modeler.createModelsFor(partitionsX, partitionsY, partitionLabels)))
        print("Creating models per partition... Done")

        # Get unseen data
        print("Reading unseen data...")
        unseenX,unseenY = dRead.UnseenSeriesReader.readSeriesDataFromFile(dRead.UnseenSeriesReader(),sFile,startU,endU,40000)
        # and their features
        unseenFeaturesX, unseenFeaturesY = featureExtractor.extractFeatures(unseenX, unseenY,future)
        print("Reading unseen data... Done")

        # Predict and evaluate on seen data
        print("Evaluating on seen data...")
        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),X,Y,modeler)
        print ("Mean absolute error on training data: %4.2f (+/- %4.2f standard error)"%(meanError, sdError/sqrt(unseenFeaturesY.shape[0])))
        print("Evaluating on seen data... Done.")

        # Predict and evaluate on unseen data
        print("Evaluating on unseen data...")
        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),unseenFeaturesX,unseenFeaturesY,modeler)
        print ("Mean absolute error on unseen data: %4.2f (+/- %4.2f standard error)"%(meanError, sdError/sqrt(unseenFeaturesY.shape[0])))
        print("Evaluating on unseen data... Done.")

        # # Evaluate performance
        # evaluator = eval.MeanAbsoluteErrorEvaluation()
        # evaluator.evaluate(X)
        errors.append(meanError)
        #clusters.append(k)
        #trSize.append(trSetlen)
        print ("Pipeline for K="+str(k)+ " Clusters and Training Size "+str(trLen)+" done.")
    plotErr =plres.ErrorGraphs()
    plotErr.ErrorGraphswithKandTrlen(errors,k,trLen,True)

def initParameters():
    sFile = "./kaklis.csv"
    # Get file name
    history = 20
    future=30
    start = 2000
    end = 12000
    startU = 30000
    endU = 30900
    if len(sys.argv) > 1:
        sFile = sys.argv[1]
        history = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
    return end, endU, history, future,sFile, start, startU


# # ENTRY POINT
if __name__ == "__main__":
    main()
#     v=np.round(np.random.rand(10, 3), 1)
#     v=np.append(v,v[1:3], axis=0)
#     r= np.dot(np.sum(v, axis=1), np.diag(np.random.rand(v.shape[0])))
#     print("Input:\n%s"%(str(np.append(v, np.reshape(r, (v.shape[0],1)), axis=1))))
#
#     clusterer = dPart.BoundedProximityClusterer()
#     print(clusterer.clustering(v, r, showPlot=False))
#
