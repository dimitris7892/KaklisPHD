import dataReading as dRead
import featureCalculation as fCalc
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import plotResults as plRes
import numpy as np
from math import sqrt
import sys
import pandas as pd
import random
from sklearn.model_selection import train_test_split

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
    plotRes = plRes.ErrorGraphs()

    #plotRes.PlotTrueVsPredLine()
    dataErrorFocDTC = pd.read_csv('/home/dimitris/Desktop/___RES/DTC.csv', delimiter=',')
    dataErrorFocKM = pd.read_csv('/home/dimitris/Desktop/resultsNEW/kmeansNEW.csv', delimiter=',')

    #plotRes.PlotExpRes(dataErrorFocDTC,'dtc')
    #plotRes.PlotExpRes(dataErrorFocKM,'kmeans')
    #plotRes.boxPLotsKMDT()
    #plotRes.boxPLots()
    #plotRes.computeMeansStd()
    #plotRes.generateGraphVRPM()


    '''results={}
    dataset = 0
    results['EstimatorResults'] = []
    #outerItem = {"estimator": , "speed": (velMin + velMax) / 2, "cells": []}
    dataRes = pd.read_csv('/home/dimitris/Desktop/LAROS/NEWres_1.csv').values
    initVar = []
    for i in range(0,len(dataRes)):
        #if (i % 33 ==0 and i > 0) :
        initVar.append(dataRes[i][4])
        if i > 0 and initVar[i]!=initVar[i-1]:
            dataset =dataset+1
        cluster = dataRes[i][0]
        error = dataRes[i][1]
        est = dataRes[i][2]
        part = dataRes[i][3]
        item = {"dataset":dataset,"estimator":est, "error": float(error), "cluster": int(cluster),'partitioner':part}
        results['EstimatorResults'].append(item)

    minErrors=[]
    minClusters=[]
    for i in range(0,5):
        listOfDict = [k for k in results['EstimatorResults'] if k['dataset'] == i and k['estimator'] == 'TensorFlowCA' and k['partitioner']=='DelaunayTriPartitioner']
        listOfErrors = [x['error'] for x in listOfDict]
        listOfClusters = [x['cluster'] for x in listOfDict]
        minIndex = listOfErrors.index(min([x['error'] for x in listOfDict]))
        minErrors.append(listOfErrors[minIndex])
        minClusters.append(listOfClusters[minIndex])
        ####

    meanError = np.mean(minErrors)
    stdError =np.std(minErrors)
    print("Mean absolute error on unseen data: %4.2f (+/- %4.2f standard error)" % (meanError, stdError / sqrt(len(minErrors))) +" Mean clusters: "+ str(np.mean(minClusters)))

    listOfDict = [k for k in results['EstimatorResults'] if  k['estimator'] == 'TensorFlowWLSTM']
    listOfErrors1 = [x['error'] for x in listOfDict]
    print(np.mean(listOfErrors))

    listOfDict = [k for k in results['EstimatorResults'] if  k['estimator'] == 'TensorFlowWLSTM2']
    listOfErrors2 = [x['error'] for x in listOfDict]
    print(np.mean(listOfErrors))'''

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
        elif al == 'NNWE':
            modelers.append(dModel.TensorFlowWeights())
        elif al == 'NNWLSTM':
            modelers.append(dModel.TensorFlowWLSTM())
        elif al == 'NNWLSTM2':
            modelers.append(dModel.TensorFlowWLSTM2())



    partitioners=[]
    for cl in cls:
        if cl=='KM':partitioners.append(dPart.KMeansPartitioner())
        if cl=='DC' :partitioners.append(dPart.DelaunayTriPartitioner())
        if cl == 'NNCL': partitioners.append(dPart.TensorFlowCl())

    print(modelers)



    meanBTr=[]
    meanVTr=[]
    #random.seed(1)
    subsetsW=[]
    unseenXDt=[]
    unseenYDt=[]
    data = pd.read_csv(sFile)
    for k in range(0,500):
        # The row indices to skip - make sure 0 is not included to keep the header!
        skip_idx = random.sample(range(1, num_linesx), num_linesx-size)
        # Read the data
                           #skiprows=skip_idx)
        #print(str(k))
        try:
            seriesX, targetY, targetW,targetB = reader.readStatDifferentSubsets(data, subsetsX, subsetsY,k, 4000)
        except:
            c=0
        if seriesX == [ ] and targetY == [ ] : continue

        '''dataset = np.array(np.append(seriesX.reshape(-1,1), np.asmatrix([targetY]).T, axis=1))

        for i in range(0, len(dataset)):
            dataset[i] = np.mean(dataset[i:i + 10], axis=0)

        seriesX = dataset[:,0]
        targetY = dataset[:, 1]'''

        X_train, X_test, y_train, y_test = train_test_split(seriesX, targetY, test_size=0.2,random_state=42)

        subsetsX.append(X_train.astype(float))
        subsetsY.append(y_train.astype(float))

        #unseenXDt.append(X_test.astype(float))
        #unseenYDt.append(y_test.astype(float))
        #subsetsX.append(seriesX.astype(float))
        #subsetsY.append(targetY.astype(float))

        unseenXDt.append(X_test.astype(float))
        unseenYDt.append(y_test.astype(float))

        #subsetsX.append(seriesX)
        #subsetsY.append(targetY)
        #subsetsW.append(targetW)
        #subsetsB.append(targetB)
        var.append(np.var(seriesX))

        if len(subsetsX)>=1:
            break

    print(subsetsX[0].shape)
    rangeSubs = k
    stdInU = [ ]
    varTr=[]
    models=[]

    part=[]
    histTr=[]
    counter=0



    K = range(1,12)
    print("Number of Statistically ind. subsets for training: " + str(len(subsetsX)))
    subsetsX=   subsetsX[0:5] if len(subsetsX) > 5 else subsetsX[0:5]
    subsetsY =  subsetsY[0:5]  if len(subsetsY) > 5 else subsetsY[0:5]
    #K=[10]
    subsetsCounter=0

    for subsetX, subsetY in zip(subsetsX, subsetsY):

      #skip_idxx = random.sample(range((rangeSubs * 1000) + 10, num_lines), 12000)
        # Read the data
      #data1 = pd.read_csv(sFile, skiprows=skip_idxx)
      for modeler in modelers:
        for partitioner in partitioners:
           if (modeler.__class__.__name__ == 'TensorFlowWLSTM2' or modeler.__class__.__name__ == 'TensorFlowWLSTM') and partitioner.__class__.__name__ == 'DelaunayTriPartitioner':
               break
           if modeler.__class__.__name__ == 'TriInterpolantModeler' and partitioner.__class__.__name__=='DelaunayTriPartitioner':
                break
           if modeler.__class__.__name__ == 'TriInterpolantModeler' or modeler.__class__.__name__ == 'TensorFlow':
                t=0
           if partitioner.__class__.__name__=='DelaunayTriPartitioner':
                 partK=[0.4]
                     #np.linspace(0.05,1.5,11)#[0.5]

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
           clustersGeneratedByDC = []
           for k in partK:

                #try:
                    if k == partK[len(partK)-1]:
                        np.append(partK, 6)
                    print(modeler.__class__.__name__)
                    print("Reading data...")
                    if (modeler.__class__.__name__ == 'TensorFlowWLSTM' or modeler.__class__.__name__ == 'TensorFlowWLSTM2') and k > 1:
                        break
                    reader = dRead.BaseSeriesReader()
                    #or modeler.__class__.__name__ == 'TensorFlowWLSTM2'
                    print("Reading data... Done.")
                    if (modeler.__class__.__name__ == 'TensorFlowWLSTM3' or modeler.__class__.__name__ == 'TensorFlowW3' ) and partitioner.__class__.__name__ != 'DelaunayTriPartitioner':

                        dataset = np.array(np.append(subsetX.reshape(-1, subsetX.shape[1] if len(subsetX.shape)>1 else 1), np.asmatrix([subsetY]).T, axis=1))

                        for i in range(0, len(dataset)):
                            dataset[i] = np.mean(dataset[i:i + 15], axis=0)

                        subsetX = dataset[:, 0:(dataset.shape[1]) - 1 if len(dataset.shape)>1 else 1]
                        subsetY = dataset[:, dataset.shape[1] - 1 if len(dataset.shape)>1 else 1]
                    # Extract features
                    seriesX = subsetX
                    targetY = subsetY
                    if modeler.__class__.__name__ != 'TensorFlowWD':
                        print("Extracting features from training set...")
                        featureExtractor = fCalc.BaseFeatureExtractor()
                        X, Y , W = featureExtractor.extractFeatures(modeler,seriesX, targetY,targetW,targetB,history)
                        '''dataset = np.array(np.append(X, np.asmatrix([Y]).T, axis=1))
    
                        for i in range(0, len(dataset)):
                            dataset[i] = np.mean(dataset[i:i + 10], axis=0)
    
                        X = dataset[:, 0:2]
                        Y = dataset[:, 2]'''


                        print("Extracting features from training set... Done.")
                    #X = subsetX
                    #Y = subsetY
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
                    if len(partitionsX) in clustersGeneratedByDC: continue
                    clustersGeneratedByDC.append(len(partitionsX))
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

                        #unseenX,unseenY , unseenW,unseenB = dRead.UnseenSeriesReader.readRandomSeriesDataFromFile(dRead.UnseenSeriesReader(),data1)
                    # and their features
                    ##
                        #unseenX=unseenX[0:2880]
                        #unseenY=unseenY[0:2880]
                        #unseenW = unseenW[ 0:2880 ]
                        unseenX = unseenXDt[subsetsCounter]
                        unseenY = unseenYDt[subsetsCounter]
                        # or modeler.__class__.__name__ == 'TensorFlowLSTM2'
                        #or modeler.__class__.__name__ == 'TensorFlowLSTM2'
                        if (modeler.__class__.__name__ == 'TensorFlowWLSTM3'  or modeler.__class__.__name__ == 'TensorFlowW3') \
                        and partitioner.__class__.__name__ != 'DelaunayTriPartitioner':

                            dataset = np.array(
                                np.append(unseenX.reshape(-1, unseenX.shape[1] if len(unseenX.shape)>1 else 1), np.asmatrix([unseenY]).T, axis=1))

                            for i in range(0, len(dataset)):
                                dataset[i] = np.mean(dataset[i:i +15], axis=0)

                            unseenX = dataset[:, 0:(dataset.shape[1]) - 1 if len(dataset.shape)>1 else 1]
                            unseenY = dataset[:, dataset.shape[1] - 1 if len(dataset.shape)>1 else 1]
                        stdInU.append(np.std(unseenX))
                    ##
                        featureExtractor=fCalc.UnseenFeaturesExtractor()
                        unseenFeaturesX, unseenFeaturesY , unseenFeaturesW = featureExtractor.extractFeatures(modeler,unseenX, unseenY,None,None,history)
                        #dataset = np.array(np.append(unseenFeaturesX, np.asmatrix([unseenFeaturesY]).T, axis=1))



                        '''for i in range(0, len(dataset)):
                            dataset[i] = np.mean(dataset[i:i + 10], axis=0)
    
                        unseenFeaturesX = dataset[:, 0:2]
                        unseenFeaturesY = dataset[:, 2]'''

                    print("Reading unseen data... Done")

                    # Predict and evaluate on seen data
                    print("Evaluating on seen data...")
                    '''if modeler.__class__.__name__  != 'TriInterpolantModeler':
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
                        print("Evaluating on seen data... Done.")'''

                    # Predict and evaluate on unseen data
                    print("Evaluating on unseen data...")
                    if modeler.__class__.__name__ != 'TensorFlowW1' and modeler.__class__.__name__ != 'TensorFlow' and modeler.__class__.__name__ != 'TensorFlowW' and modeler.__class__.__name__ != 'TriInterpolantModeler' \
                            and modeler.__class__.__name__ != 'TensorFlowWD' and  modeler.__class__.__name__ != 'TensorFlowWeights' and  modeler.__class__.__name__ != 'TensorFlowWLSTM'\
                            and  modeler.__class__.__name__ != 'TensorFlowWLSTM2':
                        Errors, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluate(eval.MeanAbsoluteErrorEvaluation(),
                                                                                          unseenFeaturesX, unseenFeaturesY, modeler,genericModel,partitionsX)


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

                    elif modeler.__class__.__name__ == 'TensorFlowWeights':
                        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNNNweights(
                            eval.MeanAbsoluteErrorEvaluation(), unseenFeaturesX,
                            unseenFeaturesY,
                            modeler, output, None, None, partitionsX, genericModel)

                    elif modeler.__class__.__name__ == 'TensorFlowWLSTM':
                        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNNLSTM(
                            eval.MeanAbsoluteErrorEvaluation(), unseenFeaturesX,
                            unseenFeaturesY,
                            modeler, output, None, None, partitionsX, genericModel)

                    elif modeler.__class__.__name__ == 'TensorFlowWLSTM2':
                        _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNNLSTM2(
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
                    if modeler.__class__.__name__ == 'TriInterpolantModeler':
                        part.append('none')
                    else:
                        part.append(partitioner.__class__.__name__)

                    errors.append(meanError)
                    '''if modeler.__class__.__name__ == 'TriInterpolantModeler':
                        trErrors.append(None)
                    else:
                        trErrors.append(meanErrorTr)'''
                    err={}
                    err["model"] =modeler.__class__.__name__
                    err["error"] = meanError
                    err["k"]=k
                    error["errors"].append(err)
                    if modeler.__class__.__name__ == 'TriInterpolantModeler' and numOfclusters==1 :
                    #or modeler.__class__.__name__ == 'TriInterpolantModeler' \
                            #and partitioner.__class__.__name__ == 'DelaunayTriPartitioner':
                        break
                    #if partitioner.__class__.__name__ == 'DelaunayTriPartitioner' and numOfclusters==1:
                        #break
                #except Exception as e:
                    #print(str(e))
                    #print(str(modeler.__class__.__name__ +" "+str(partitioner.__class__.__name__ )))
                    #eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr,
                                                               #trErrors, errors, models, part)
                    eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr,
                                                                       trErrors, errors, models, part)
      subsetsCounter = subsetsCounter + 1



    eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr,trErrors ,errors,models,part)

def initParameters():
    sFile = "./kaklis.csv"

    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000

    algs=['NNWLSTM']
        #['SR','LR','RF','NNW','NNW1','NNWCA']
        #`['SR','LR','RF','NNW','NNW1','NNWCA','NNWE','NNWLSTM','NNWLSTM2']`
    #algs= ['SR','LR','RF','NNW','NNW1','NNWCA','NNWE','NNWLSTM']
        #['SR','LR','RF','NNW','NNW1','NNWCA','TRI']
    # ['SR','LR','RF','NN','NNW','TRI']


        #['SR','LR','RF','NN','NNW','TRI']
    cls=['KM']
        #['KM','DC','NNCL']
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

