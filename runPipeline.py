import dataReading as dRead
import featureCalculation as fCalc
import dataPartitioning as dPart
import dataModeling as dModel
# import evaluation as eval
import sys

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
partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel = partitioner.clusteringV(X, Y, NUM_OF_CLUSTERS, True)
# Keep label to partition mapping in a dict
partitionXPerLabel = dict(zip(partitionLabels, partitionsX))
partitionYPerLabel = dict(zip(partitionLabels, partitionsY))

# For each partition create model
modeler = dModel.LinearRegressionModeler()
# ...and keep them in a dict, connecting label to model
modelMap = dict(zip(partitionLabels, modeler.createModelsFor(partitionsX, partitionsY, partitionLabels)))

# Get unseen data
unseenDataSource = dRead

print ("Pipeline done.")