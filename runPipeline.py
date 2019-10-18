import dataReading as dRead
import featureCalculation as fCalc
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import numpy as np
from math import sqrt
import sys
import scipy.stats as st

def main():
    # Init parameters based on command line
    end, endU, history, sFile, start, startU = initParameters()
    # Load data
    print("Reading data...")
    reader = dRead.BaseSeriesReader()

    seriesX, targetY ,targetW= reader.readSeriesDataFromFile(sFile,start,end,2000)
    print("Reading data... Done.")

    # Extract features
    print("Extracting features from training set...")
    featureExtractor = fCalc.BaseFeatureExtractor()
    X, Y ,W = featureExtractor.extractFeatures(seriesX, targetY,targetW,history)
    print("Extracting features from training set... Done.")

    # Partition data
    print("Partitioning training set...")
    NUM_OF_CLUSTERS=1# TODO: Read from command line

    # partitioner = dPart.DefaultPartitioner()
    partitioner = dPart.KMeansPartitioner()
    #partitioner = dPart.BoundedProximityPartitioner()
    partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel = partitioner.clustering(X, Y,W, NUM_OF_CLUSTERS, False)
    # Keep label to partition mapping in a dict
    partitionXPerLabel = dict(zip(partitionLabels, partitionsX))
    partitionYPerLabel = dict(zip(partitionLabels, partitionsY))
    print("Partitioning training set... Done.")

    # For each partition create model
    print("Creating models per partition...")
    modeler = dModel.TensorFlow()
    # ...and keep them in a dict, connecting label to model
    modelMap, xs, output = modeler.createModelsFor(partitionsX, partitionsY, partitionLabels)
    modelMap = dict(zip(partitionLabels, modelMap))
    print("Creating models per partition... Done")

    # Get unseen data
    print("Reading unseen data...")
    unseenX,unseenY,unseenW = dRead.UnseenSeriesReader.readSeriesDataFromFile(dRead.UnseenSeriesReader(),sFile,startU,endU,40000)
    # and their features
    unseenFeaturesX, unseenFeaturesY,unseenFeaturesW = featureExtractor.extractFeatures(unseenX, unseenY,unseenW,history)
    print("Reading unseen data... Done")

    # Predict and evaluate on seen data
    print("Evaluating on seen data...")
    if modeler.__class__.__name__ != 'TensorFlow':
        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(), X, Y,
                                                                          modeler)
    else:
        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateNN(eval.MeanAbsoluteErrorEvaluation(), X,
                                                                            Y,
                                                                            modeler, output, xs)
    print ("Mean absolute error on training data: %4.2f (+/- %4.2f standard error)" % (
    meanError, sdError / sqrt(unseenFeaturesY.shape[ 0 ])))
    print("Evaluating on seen data... Done.")

    # Predict and evaluate on unseen data
    print("Evaluating on unseen data...")
    if modeler.__class__.__name__ != 'TensorFlow':
        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),
                                                                          unseenFeaturesX, unseenFeaturesY, modeler)
    else:
        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateNN(eval.MeanAbsoluteErrorEvaluation(),
                                                                            unseenFeaturesX, unseenFeaturesY,
                                                                            modeler, output, xs)


    print("Evaluating on unseen data... Done.")

    print ("Mean absolute error on unseen data: %4.2f (+/- %4.2f standard error)"%(meanError, sdError/sqrt(unseenFeaturesY.shape[0])))
    print("Evaluating on unseen data... Done.")
    print("Standard Deviation of training Data: %4.2f" %(np.std(X)))
    print("Standard Deviation of unseen Data: %4.2f" % (np.std(unseenX)))
    print("Percentage of variance in Training data explained in Unseen Dataset: %4.2f" %(np.var(X)/np.var(unseenX))/100+"%")

    # # Evaluate performance
    # evaluator = eval.MeanAbsoluteErrorEvaluation()
    # evaluator.evaluate(X)

    print ("Pipeline done.")


def initParameters():
    sFile = "./kaklis.csv"
    # Get file name
    history = 20
    start = 11000
    end = 15000
    startU = 4000
    endU = 5000
    if len(sys.argv) > 1:
        sFile = sys.argv[1]
        history = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
    return end, endU, history, sFile, start, startU


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
