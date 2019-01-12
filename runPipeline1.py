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
    K=[np.square(i) for i  in range(1,10)]
    #trSetlen=[i for i in range(1000,30000,1000)]
    trSetlen=[500,1000,2000,3000,5000,10000,20000,30000,40000]
    errors=[]
    clusters=[]
    trSize=[]
    #KtrSetlen=zip(K,trSetlen)
    #KtrSetlen=list(itertools.product(K, trSetlen))

    modeler = dModel.SplineRegressionModeler()
    cutoff=np.linspace(0.1,1,111)
    #for k,trLen in zip(K,trSetlen):
    minErr=[]
    minK=[]
    #for trLen in trSetlen:
      #errors=[]
    for k in [1]:
        print("Reading data...")
        reader = dRead.BaseSeriesReader()
        trSize=80000
        seriesX, targetY, targetW = reader.readSeriesDataFromFile(sFile,start,end,trSize,'t')
        print("Reading data... Done.")

        # Extract features
        print("Extracting features from training set...")
        featureExtractor = fCalc.BaseFeatureExtractor()
        X, Y , W = featureExtractor.extractFeatures(seriesX, targetY,targetW,history)
        print("Extracting features from training set... Done.")

        # Partition data
        print("Partitioning training set...")
        NUM_OF_CLUSTERS =k# TODO: Read from command line
        NUM_OF_FOLDS=6
        #partitioner = dPart.DefaultPartitioner()
        partitioner = dPart.CrossValidationPartioner()
        #partitioner = dPart.BoundedProximityPartitioner()
        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel , partitionsXt,partitionsYt = partitioner.NfoldValidation(X, Y,W,NUM_OF_FOLDS, False)
        # Keep label to partition mapping in a dict
        #partitionXPerLabel = dict(zip(partitionLabels, partitionsX))
        #partitionYPerLabel = dict(zip(partitionLabels, partitionsY))
        print("Partitioning training set... Done.")

        # For each partition create model
        print("Creating models per partition...")

        # ...and keep them in a dict, connecting label to model
        modelMap = dict(zip(partitionLabels, modeler.createModelsFor(partitionsX, partitionsY, partitionLabels)))
        print("Creating models per partition... Done")

        # Get unseen data
        print("Reading unseen data...")
        unseenX,unseenY , unseenW = dRead.UnseenSeriesReader.readSeriesDataFromFile(dRead.UnseenSeriesReader(),sFile,startU,endU,40000,'u')
        # and their features
        featureExtractor=fCalc.UnseenFeaturesExtractor()
        unseenFeaturesX, unseenFeaturesY , unseenFeaturesW = featureExtractor.extractFeatures(unseenX, unseenY,unseenW,history)
        print("Reading unseen data... Done")

        # Predict and evaluate on seen data
        print("Evaluating on seen data...")
        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),np.concatenate(partitionsXt),np.concatenate(partitionsYt),modeler)
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
        clusters.append(len(partitionLabels))
        #clusters.append(k)
        #trSize.append(trLen)
        #print ("Pipeline for K="+str(k)+ " Clusters and Training Size "+str(trLen)+" done.")
      #minerr=min(errors)
      #mink=K[errors.index(min(errors))]
      ###################################
      #minErr.append(minerr)
      #minK.append(mink)
      ###################################
    optcl = clusters[ errors.index(min(errors)) ]
    plotErr =plres.ErrorGraphs()
    plotErr.ErrorGraphsForPartioners(errors,cutoff,3000,True,modeler.__class__.__name__,partitioner.__class__.__name__)
    #plotErr.ErrorGraphswithKandTrlen(errors, 6000, trSetlen, True, modeler.__class__.__name__)
    #print ("Min Error with "+str(modeler.__class__.__name__)+" "+ np.min(errors))
def initParameters():
    sFile = "./kaklis.csv"
    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000
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
