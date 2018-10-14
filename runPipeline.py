import dataReading as dRead
import featureCalculation as fCalc
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import sys

# Get file name
sFile = "./kaklis.csv"
history=20
start=2000
end=12000
startU=30000
endU=30900
if len(sys.argv) > 1:
    sFile = sys.argv[1]
    history = sys.argv[2]
    start = sys.argv[3]
    end =sys.argv[4]
# Load data
reader = dRead.BaseSeriesReader()
seriesX, targetY = reader.readSeriesDataFromFile(sFile,start,end,20000)

# Extract features
featureExtractor = fCalc.BaseFeatureExtractor()
X, Y = featureExtractor.extractFeatures(seriesX, targetY,history)

# Partition data
partitioner = dPart.KMeansClusterer()
NUM_OF_CLUSTERS = 30
partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel = partitioner.clusteringV(X, Y, NUM_OF_CLUSTERS, False)
# Keep label to partition mapping in a dict
partitionXPerLabel = dict(zip(partitionLabels, partitionsX))
partitionYPerLabel = dict(zip(partitionLabels, partitionsY))

# For each partition create model
modeler = dModel.LinearRegressionModeler()
# ...and keep them in a dict, connecting label to model
modelMap = dict(zip(partitionLabels, modeler.createModelsFor(partitionsX, partitionsY, partitionLabels)))

# Get unseen data
unseenX,unseenY = dRead.UnseenSeriesReader.readSeriesDataFromFile(dRead.UnseenSeriesReader(),sFile,startU,endU,40000)

#Predict
error = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),unseenX,unseenY,modeler)
print error

print ("Pipeline done.")