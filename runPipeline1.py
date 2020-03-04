import dataReading as dRead
#from Danaos_ML_Project import dataReading as DANdRead
import  Danaos_ML_Project.dataReading as DANdRead
import featureCalculation as fCalc
import dataReadingD as DANRead
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import numpy as np
from math import sqrt
import sys
import plotResults as plres
import itertools
import pandas as pd
import random
from sklearn.decomposition import PCA
import datetime


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
    var=[]
    clusters=[]
    cutoff=np.linspace(0.1,1,111)
    subsetsX=[]
    subsetsY = []
    subsetsB=[]
    reader = dRead.BaseSeriesReader()

    DANreader = DANdRead.BaseSeriesReader()
    #DANreader.GenericParserForDataExtraction('LAROS', 'MARMARAS', 'MT_DELTA_MARIA')
    #DANreader = DANRead.BaseSeriesReader()
    #DANreader.readLarosDAta(datetime.datetime(2018,1,1),datetime.datetime(2019,1,1))

    #DANreader.GenericParserForDataExtraction('LAROS','MARMARAS','MT_DELTA_MARIA')
    #DANreader.GenericParserForDataExtraction('TELEGRAMS', 'GOLDENPORT', 'TRAMMO LAOURA',driver='ORACLE',server='10.2.5.80',sid='OR12',usr='goldenport',password='goldenport',pathOfRawData='C:/Users/dkaklis/Desktop/danaos')
    #DANreader.readExtractNewDataset('MILLENIA','FANTASIA',';')
    #return
    #DANreader.ExtractLAROSDataset("",'2017-06-01 00:00:00','2019-10-09 15:10:00')
    #return
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
        elif al == 'NNWD':
            modelers.append(dModel.TensorFlowWD())

    partitioners=[]
    for cl in cls:
        if cl=='KM':partitioners.append(dPart.KMeansPartitioner())
        if cl=='DC' :partitioners.append(dPart.DelaunayTriPartitioner())

    print(modelers)


    data = pd.read_csv(sFile)
    meanBTr=[]
    meanVTr=[]
    #random.seed(1)
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

        if len(subsetsX)>=5:

            break
    #subsetsX=[subsetsX[0]]
    #subsetsY=[subsetsY[0]]
    rangeSubs = k
    stdInU = [ ]
    varTr=[]
    models=[]
    part=[]
    histTr=[]
    counter=0

    K = range(1,12)
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
           if partitioner.__class__.__name__=='DelaunayTriPartitioner':
                 partK=[0.5]
                 #np.linspace(0.1,1,11)
                     #[0.6]
           if partitioner.__class__.__name__=='KMeansPartitioner':
               if modeler.__class__.__name__=='TriInterpolantModeler' or modeler.__class__.__name__ == 'TensorFlow':
                 partK =K
               else:
                 partK=K
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
                    #reader.insertDataAtDb()
                    #reader.readNewDataset()
                    reader.readExtractNewDataset('MILLENIA')
                    #data = pd.read_csv('./MT_DELTA_MARIA_data_1.csv')
                    #reader.readLarosDataFromCsvNewExtractExcels(data)
                    seriesX, targetY,unseenFeaturesX, unseenFeaturesY  , drftB6 , drftS6 , drftTargetB6 , drftTargetS6, partitionsX, partitionsY,partitionLabels = reader.readLarosDataFromCsvNew(data)
                #################

                #seriesX, targetY ,targetW= reader.readStatDifferentSubsets(data,subsetsX,subsetsY,2880)
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

                #partitionsX, partitionsY , partitionLabels=X,Y,W
                #if modeler.__class__.__name__!='TensorFlow':
                # Partition data
                print("Partitioning training set...")
                NUM_OF_CLUSTERS =k# TODO: Read from command line
                NUM_OF_FOLDS=6
                #if modeler!='TRI':
                if modeler.__class__.__name__ != 'TensorFlowWD':
                    partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel  , centroids  = partitioner.clustering(X, Y, W ,NUM_OF_CLUSTERS, True,k)
                else:
                   #partitionLabels=23

                    partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel, tri  = partitioner.clustering(

                        seriesX, targetY, None, NUM_OF_CLUSTERS, True, k)

                    #partitionsXDB6, partitionsYDB6, partitionLabels, partitionRepresentatives, partitioningModel, tri = partitioner.clustering(
                        #drftB6, targetY, None, 25, True, k)

                    #partitionsXDBS6, partitionsYDS6, partitionLabels, partitionRepresentatives, partitioningModel, tri = partitioner.clustering(
                        #drftS6, targetY, None, 25, True, k)

                print("Partitioning training set... Done.")
                # For each partition create model
                print("Creating models per partition...")

                #if modeler.__class__.__name__ == 'TensorFlowWD':
                    #X = np.array(np.concatenate(partitionsX))
                    #Y = np.array(np.concatenate(partitionsY))
                #skip_idx1 = random.sample(range(num_linesx, num_lines), (num_lines-num_linesx) - 1000)

                #modeler.plotRegressionLine(partitionsX, partitionsY, partitionLabels,genericModel,modelMap)
                # ...and keep them in a dict, connecting label to model
                #modelMap, xs, output, genericModel =None,None,None,None
                unseenX=[]
                unseenY=[]
                if modeler.__class__.__name__!= 'TriInterpolantModeler' :
                            #and modeler.__class__.__name__ != 'TensorFlow':
                    modelMap, model2,scores, output,genericModel = modeler.createModelsFor(partitionsX, partitionsY, partitionLabels,None,X,Y)
                            #, genericModel , partitionsXDC
                    #if modeler.__class__.__name__ != 'TensorFlow':
                        #modelMap = dict(zip(partitionLabels, modelMap))
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
                if modeler.__class__.__name__  != 'TriInterpolantModeler':
                    print("Evaluating on seen data...")

                    if modeler.__class__.__name__ != 'TensorFlow'and modeler.__class__.__name__ != 'TensorFlowW' and modeler.__class__.__name__ != 'TriInterpolantModeler' and modeler.__class__.__name__ != 'TensorFlowWD':
                        x=1
                        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(), X,
                                                                                          Y,
                                                                                          modeler, genericModel)


                    elif modeler.__class__.__name__ == 'TensorFlow' or modeler.__class__.__name__ == 'TensorFlowW' or modeler.__class__.__name__ == 'TensorFlowWD':
                        #_,meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(
                            #eval.MeanAbsoluteErrorEvaluation(), X,
                            #Y,
                            #modeler, output, xs,genericModel,partitionsXDC)
                            x=1

                    #print ("Mean absolute error on training data: %4.2f (+/- %4.2f standard error)" % (
                        #meanError, sdError / sqrt(unseenFeaturesY.shape[ 0 ])))
                    #print("Evaluating on seen data... Done.")

                # Predict and evaluate on unseen data
                print("Evaluating on unseen data...")
                if modeler.__class__.__name__ != 'TensorFlow' and modeler.__class__.__name__ != 'TensorFlowW' and modeler.__class__.__name__ != 'TriInterpolantModeler' and modeler.__class__.__name__ != 'TensorFlowWD':
                    Errors, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),
                                                                                      unseenFeaturesX, unseenFeaturesY, modeler,genericModel)


                elif modeler.__class__.__name__ == 'TensorFlow' or modeler.__class__.__name__ == 'TensorFlowW' or  modeler.__class__.__name__ == 'TensorFlowWD':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(
                        eval.MeanAbsoluteErrorEvaluation(),unseenFeaturesX,
                        unseenFeaturesY,
                        modeler,output, None,None,partitionsX , genericModel)
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
                #varExpl=np.sum(np.square(subsetX-unseenX)) / np.square(np.sum(subsetX)-np.sum(unseenX))*100
                #print("Percentage of variance in Training data explained in Unseen Dataset: %4.2f" % (
                        #(varExpl)) + " %")
                # # Evaluate performance
                numOfclusters= len(partitionsX)
                if partitioner.__class__.__name__ == 'DelaunayTriPartitioner' and numOfclusters==1:
                    break
                clusters.append(numOfclusters)
                varTr.append(np.var(subsetX))
                if modeler.__class__.__name__  != 'TriInterpolantModeler':
                    models.append(modeler.__class__.__name__)
                else:
                    models.append('Analytical method')
                part.append(partitioner.__class__.__name__)
                #meanVTr.append(np.mean(s ubsetX))
                #meanBTr.append(np.mean(X[:,2]))
                errors.append(meanError)


                err={}
                err["model"] =modeler.__class__.__name__
                err["error"] = meanError
                err["k"]=k
                error["errors"].append(err)
                if modeler.__class__.__name__ == 'TriInterpolantModeler' and numOfclusters==1:
                    break


    eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr, errors,models,part)

def initParameters():
    sFile = "./kaklis.csv"
    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000
    algs=['NNW']
    # ['SR','LR','RF','NN','NNW','TRI']


        #['SR','LR','RF','NN','NNW','TRI']
    cls=['KM']
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
#     v=np.round(np.random.rand(10, 3), 1)
#     v=np.append(v,v[1:3], axis=0)
#     r= np.dot(np.sum(v, axis=1), np.diag(np.random.rand(v.shape[0])))
#     print("Input:\n%s"%(str(np.append(v, np.reshape(r, (v.shape[0],1)), axis=1))))
#
#     clusterer = dPart.BoundedProximityClusterer()
#     print(clusterer.clustering(v, r, showPlot=False))
#
