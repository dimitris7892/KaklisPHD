import dataReading as dRead
import featureCalculation as fCalc
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import numpy as np
from math import sqrt
import sys
import pandas as pd
import random

def main():
    # Init parameters based on command line

    end, endU, history, future,sFile, start, startU , algs , cls = initParameters()
    # Load data
    if len(sys.argv) >1:
        algs = algs.split(',')
        cls=cls.split(',')
        print(sFile , algs , cls)
        end = int(end)
        endU = int(endU)
        history = int(history)
        future = int(future)
        start = int(start)
        startU = int(startU)

    trSetlen=[500,1000,2000,3000,5000,10000,20000,30000,40000]
    errors=[]
    trErrors=[]
    var=[]
    clusters=[]
    cutoff=np.linspace(0.1,1,111)
    subsetsX=[]
    subsetsY = []
    subsetsB=[]
    reader = dRead.BaseSeriesReader()


    ####
    num_lines = sum(1 for l in open(sFile))
    num_linesx = num_lines

    # Sample size - in this case ~10%
    size = 5000
    modelers=[]
    for al in algs:
        if al=='SR' : modelers.append(dModel.SplineRegressionModeler())
        elif al=='LR':modelers.append(dModel.LinearRegressionModeler())
        elif al=='RF':modelers.append(dModel.RandomForestModeler())
        elif al=='NN' :modelers.append(dModel.TensorFlow())
        elif al=='TRI' : modelers.append(dModel.TriInterpolantModeler())
        elif al == 'NNW':modelers.append(dModel.TensorFlowW())
        elif al == 'NNWD':modelers.append(dModel.TensorFlowWD())
        elif al == 'NNW1':modelers.append(dModel.TensorFlowW1())
        elif al == 'NNWCA':modelers.append(dModel.TensorFlowCA())

    partitioners=[]
    for cl in cls:
        if cl=='KM':partitioners.append(dPart.KMeansPartitioner())
        if cl=='DC' :partitioners.append(dPart.DelaunayTriPartitioner())
        if cl == 'NNCL': partitioners.append(dPart.TensorFlowCl())

    print(modelers)


    data = pd.read_csv(sFile)
    meanBTr=[]
    meanVTr=[]
    random.seed(1)
    subsetsW=[]
    data = pd.read_csv(sFile)
    for k in range(0,200):
        # The row indices to skip - make sure 0 is not included to keep the header!
        skip_idx = random.sample(range(1, num_linesx), num_linesx-size)
        # Read the data
                           #skiprows=skip_idx)
        seriesX, targetY, targetW,targetB = reader.readStatDifferentSubsets(data, subsetsX, subsetsY,k, 4000)
        if seriesX == [ ] and targetY == [ ] : continue
        subsetsX.append(seriesX)
        subsetsY.append(targetY)
        subsetsW.append(targetW)
        subsetsB.append(targetB)
        var.append(np.var(seriesX))

        if len(subsetsX)>=1:
            break

    rangeSubs = k
    stdInU = [ ]
    varTr=[]
    models=[]
    part=[]
    histTr=[]
    counter=0

    K = range(1,21)
    print("Number of Statistically ind. subsets for training: " + str(len(subsetsX)))
    subsetsX=[subsetsX[0:5]] if len(subsetsX) > 5 else subsetsX
    subsetsY = [ subsetsY[ 0:5 ] ] if len(subsetsY) > 5 else subsetsY
    #K=[10]

    for subsetX, subsetY in zip(subsetsX, subsetsY):
      skip_idxx = random.sample(range((rangeSubs * 1000) + 10, num_lines), 12000)
        # Read the data

      data1 = pd.read_csv(sFile, skiprows=skip_idxx)
      for modeler in modelers:
        for partitioner in partitioners:
           if modeler.__class__.__name__ == 'TriInterpolantModeler' and partitioner.__class__.__name__=='DelaunayTriPartitioner':
                break
           if modeler.__class__.__name__ == 'TriInterpolantModeler' or modeler.__class__.__name__ == 'TensorFlow':
                partK = [1]
                partitioner.__class__.__name__ ="None"
           if partitioner.__class__.__name__=='DelaunayTriPartitioner':

                 partK=np.linspace(0.4,1,20)#[0.5]

           elif partitioner.__class__.__name__=='KMeansPartitioner':
               if modeler.__class__.__name__=='TriInterpolantModeler' or modeler.__class__.__name__ == 'TensorFlow':
                 partK =[1]
               else:
                 partK=K
           else:
               partK=[1]
           error = {"errors": [ ]}
           #random.seed(1)

           flagEvalTri = False
           for k in partK:
                if modeler.__class__.__name__ == 'TensorFlowW' and K==1: continue

                #if  modeler.__class__.__name__ == 'TensorFlow' and k>1: continue
                print(modeler.__class__.__name__)
                print("Reading data...")
                reader = dRead.BaseSeriesReader()
                trSize=80000

                ####################################LAROS DATA STATISTICAL TESTS
                if modeler.__class__.__name__ == 'TensorFlowWD':

                    reader.readExtractNewDataset('MILLENIA')
                    seriesX, targetY,unseenFeaturesX, unseenFeaturesY  , drftB6 , drftS6 , drftTargetB6 , drftTargetS6, partitionsX, partitionsY,partitionLabels = reader.readLarosDataFromCsvNew(data)
                #################

                if modeler.__class__.__name__ != 'TensorFlowWD':
                    seriesX, targetY, targetW,targetB = subsetX,subsetY,subsetsW[0],subsetsB[0]
                counter=+1

                print("Reading data... Done.")

                # Extract features
                if modeler.__class__.__name__ != 'TensorFlowWD':
                    print("Extracting features from training set...")
                    featureExtractor = fCalc.BaseFeatureExtractor()
                    X, Y , W = featureExtractor.extractFeatures(modeler,seriesX, targetY,targetW,targetB,history)
                    print("Extracting features from training set... Done.")

                print("Partitioning training set...")
                NUM_OF_CLUSTERS =k# TODO: Read from command line

                ##IF MODEL != OF ANALYTICAL METHOD IMPLEMENT CLUSTERING
                if modeler.__class__.__name__ != 'TriInterpolantModeler':
                    if modeler.__class__.__name__ != 'TensorFlowWD':
                        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel  , centroids  = partitioner.clustering(X, Y, W ,NUM_OF_CLUSTERS, True,k)
                    else:

                        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel, tri  = partitioner.clustering(

                            seriesX, targetY, None, NUM_OF_CLUSTERS, True, k)
                else:
                    partitionsX, partitionsY, partitionLabels,partitionRepresentatives,partitioningModel  , centroids= X,Y,[1],None,None,None


                print("Partitioning training set... Done.")
                # For each partition create model
                print("Creating models per partition...")


                unseenX=[]
                unseenY=[]
                if modeler.__class__.__name__!= 'TriInterpolantModeler' :

                    modelMap, history,scores, output,genericModel = modeler.createModelsFor(partitionsX, partitionsY, partitionLabels,None,X,Y)

                    print("Creating models per partition... Done")

                # Get unseen data
                print("Reading unseen data...")
                if modeler.__class__.__name__ != 'TensorFlowWD':

                    unseenX,unseenY , unseenW,unseenB = dRead.UnseenSeriesReader.readRandomSeriesDataFromFile(dRead.UnseenSeriesReader(),data1)
                # and their features
                ##
                    unseenX=unseenX[0:2880]
                    unseenY=unseenY[0:2880]
                    unseenW = unseenW[ 0:2880 ]
                    stdInU.append(np.std(unseenX))
                ##
                    featureExtractor=fCalc.UnseenFeaturesExtractor()
                    unseenFeaturesX, unseenFeaturesY , unseenFeaturesW = featureExtractor.extractFeatures(modeler,unseenX, unseenY,unseenW,unseenB,history)
                print("Reading unseen data... Done")

                # Predict and evaluate on seen data
                print("Evaluating on seen data...")
                if modeler.__class__.__name__  != 'TriInterpolantModeler':
                    #print("Evaluating on seen data...")

                    if modeler.__class__.__name__ != 'TensorFlowW1' and  modeler.__class__.__name__ != 'TensorFlow'and modeler.__class__.__name__ != 'TensorFlowW' and modeler.__class__.__name__ != 'TriInterpolantModeler' and modeler.__class__.__name__ != 'TensorFlowWD':
                        x=1
                        _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(), X,
                                                                                          Y,
                                                                                          modeler, genericModel)

                    elif modeler.__class__.__name__ == 'TensorFlow' or modeler.__class__.__name__ == 'TensorFlowW' or modeler.__class__.__name__ == 'TensorFlowWD':
                        _,meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(
                            eval.MeanAbsoluteErrorEvaluation(), X,
                            Y,
                            modeler, output, None,genericModel,partitionsX,None)

                    elif modeler.__class__.__name__ == 'TensorFlowW1':
                           _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                               eval.MeanAbsoluteErrorEvaluation(), X,
                               Y,
                               modeler, output, None,genericModel,partitionsX,None)

                    print ("Mean absolute error on training data: %4.2f (+/- %4.2f standard error)" % (
                        meanErrorTr, sdError / sqrt(unseenFeaturesY.shape[ 0 ])))
                    print("Evaluating on seen data... Done.")

                # Predict and evaluate on unseen data
                print("Evaluating on unseen data...")
                if modeler.__class__.__name__ != 'TensorFlowW1' and modeler.__class__.__name__ != 'TensorFlow' and modeler.__class__.__name__ != 'TensorFlowW' and modeler.__class__.__name__ != 'TriInterpolantModeler' and modeler.__class__.__name__ != 'TensorFlowWD':
                    Errors, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),
                                                                                      unseenFeaturesX, unseenFeaturesY, modeler,genericModel)


                elif modeler.__class__.__name__ == 'TensorFlow' or modeler.__class__.__name__ == 'TensorFlowW' or  modeler.__class__.__name__ == 'TensorFlowWD':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(
                        eval.MeanAbsoluteErrorEvaluation(),unseenFeaturesX,
                        unseenFeaturesY,
                        modeler,output, None,None,partitionsX , genericModel)
                elif modeler.__class__.__name__ == 'TensorFlowW1':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                        eval.MeanAbsoluteErrorEvaluation(), unseenFeaturesX,
                        unseenFeaturesY,
                        modeler, output, None, None, partitionsX, genericModel)
                else:
                    if flagEvalTri == False:
                        Errors, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateTriInterpolant(
                        eval.MeanAbsoluteErrorEvaluation(),
                        unseenFeaturesX, unseenFeaturesY, partitionsX, partitionsY, partitionRepresentatives,
                        partitioningModel, None, modelers[0])

                        flagEvalTri = True


                print ("Mean absolute error on unseen data: %4.2f (+/- %4.2f standard error)"%(meanError, sdError/sqrt(unseenFeaturesY.shape[0])))
                print("Evaluating on unseen data... Done.")

                print("Standard Deviation of training Data: %4.2f" % (np.std(X)))
                print("Standard Deviation of unseen Data: %4.2f" % (np.std(unseenX)))
                # # Evaluate performance
                numOfclusters= len(partitionsX)
                #pltRes = plotRes()
                #pltRes

                clusters.append(numOfclusters)
                varTr.append(np.var(subsetX))
                if modeler.__class__.__name__  != 'TriInterpolantModeler':
                    models.append(modeler.__class__.__name__)
                else:
                    models.append('Analytical method')
                part.append(partitioner.__class__.__name__)

                errors.append(meanError)
                if modeler.__class__.__name__ == 'TriInterpolantModeler':
                    trErrors.append(None)
                else:
                    trErrors.append(meanErrorTr)
                err={}
                err["model"] =modeler.__class__.__name__
                err["error"] = meanError
                err["k"]=k
                error["errors"].append(err)
                if modeler.__class__.__name__ == 'TriInterpolantModeler' and numOfclusters==1 or modeler.__class__.__name__ == 'TriInterpolantModeler' \
                        and partitioner.__class__.__name__ == 'DelaunayTriPartitioner':
                    break
                if partitioner.__class__.__name__ == 'DelaunayTriPartitioner' and numOfclusters==1:
                    break



    eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr,trErrors ,errors,models,part)

def initParameters():
    sFile = "./kaklis.csv"
    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000

    algs=['SR']
    # ['SR','LR','RF','NN','NNW','TRI']


        #['SR','LR','RF','NN','NNW','TRI']
    cls=['DC']
    #['SR','LR','RF','NN'] algs
    #['KM','DC'] clusterers / cls

    if len(sys.argv) > 1:
        sFile = sys.argv[1]
        history = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
        algs=sys.argv[5]
        cls=sys.argv[6]
    return end, endU, history, future,sFile, start, startU , algs , cls


# # ENTRY POINT
if __name__ == "__main__":
    main()

