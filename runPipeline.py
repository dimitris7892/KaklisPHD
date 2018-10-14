import dataReading as dRead
import featureCalculation as fCalc
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import numpy as np
import sys

def main():
    # Get file name
    sFile = "./kaklis_autoregr_rpm_speed.csv"
    if len(sys.argv) > 1:
        sFile = sys.argv[1]
    # Load data
    reader = dRead.BaseSeriesReader()
    seriesX, targetY = reader.readSeriesDataFromFile(sFile)

    # Extract features
    featureExtractor = fCalc.BaseFeatureExtractor()
    X, Y = featureExtractor.extractFeatures(seriesX, targetY)

    # Partition data
    partitioner = dPart.KMeansClusterer()
    NUM_OF_CLUSTERS = 30
    partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel = partitioner.clustering(X, Y, NUM_OF_CLUSTERS, True)
    # Keep label to partition mapping in a dict
    partitionXPerLabel = dict(zip(partitionLabels, partitionsX))
    partitionYPerLabel = dict(zip(partitionLabels, partitionsY))

    # For each partition create model
    modeler = dModel.LinearRegressionModeler()
    # ...and keep them in a dict, connecting label to model
    modelMap = dict(zip(partitionLabels, modeler.createModelsFor(partitionsX, partitionsY, partitionLabels)))

    # Get unseen data
    # unseenDataSource = dRead

    # Evaluate performance
    # evaluator = eval.MeanAbsoluteErrorEvaluation()
    # evaluator.evaluate(X)

    print ("Pipeline done.")

# ENTRY POINT
if __name__ == "__main__":
    # main()
    v=np.round(np.random.rand(10, 3), 1)
    v=np.append(v,v[1:3], axis=0)
    r= np.dot(np.sum(v, axis=1), np.diag(np.random.rand(v.shape[0])))
    print("Input:\n%s"%(str(np.append(v, np.reshape(r, (v.shape[0],1)), axis=1))))

    clusterer = dPart.BoundedProximityClusterer()
    print(clusterer.clustering(v, r, showPlot=False))

