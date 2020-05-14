import glob, os
import dataReading as dRead
#from Danaos_ML_Project import dataReading as DANdRead ##   NEWWWW
import dataReading as DANdRead
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
import numpy as np
from math import sqrt
import sys
import plotResults as plotRes
import itertools
import pandas as pd
import random
from sklearn.decomposition import PCA
from pylab import *
import datetime
import  matplotlib.pyplot as plt
import csv
import tensorflow as tf

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
    #DANreader.GenericParserForDataExtraction('LEMAG', 'MILLENIA', 'FANTASIA',driver='ORACLE',server='10.2.5.80',sid='OR11',usr='millenia',password='millenia',
                                             #rawData=[],telegrams=True,companyTelegrams=False,pathOfRawData='C:/Users/dkaklis/Desktop/danaos')

    #DANreader.GenericParserForDataExtraction('LEMAG', 'OCEAN_GOLD', 'PENELOPE',driver='ORACLE',server='10.2.5.80',sid='OR11',usr='oceangold',password='oceangold',
        #rawData=True,telegrams=True,companyTelegrams=True,seperator='\t',pathOfRawData='C:/Users/dkaklis/Desktop/danaos')

    #DANreader.GenericParserForDataExtraction('LEMAG', 'MARMARAS', 'MT_DELTA_MARIA',driver='ORACLE',server='10.2.5.80',sid='OR11',usr='oceangold',password='oceangold',
    #rawData=True,telegrams=False,companyTelegrams=False,seperator='\t',pathOfRawData='C:/Users/dkaklis/Desktop/danaos')

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
        elif al == 'NNWD':modelers.append(dModel.TensorFlowWD())
        elif al == 'NNW1':modelers.append(dModel.TensorFlowW1())
        elif al == 'NNW2':
            modelers.append(dModel.TensorFlowW2())
        elif al == 'NNW3':
            modelers.append(dModel.TensorFlowW3())
        elif al == 'NNWCA':modelers.append(dModel.TensorFlowCA())
        elif al == 'LI':
            modelers.append(dModel.PavlosInterpolation())

    partitioners=[]
    for cl in cls:
        if cl=='KM':partitioners.append(dPart.KMeansPartitioner())
        if cl == 'KMWSWA': partitioners.append(dPart.KMeansPartitionerWS_WA())
        if cl == 'KMWHWD': partitioners.append(dPart.KMeansPartitionerWH_WD())
        if cl=='DC' :partitioners.append(dPart.DelaunayTriPartitioner())

    print(modelers)


    #random.seed(1)
    stdInU=[]

    #dataV = pd.read_csv('/home/dimitris/Desktop/errorSTW25.csv', delimiter=',')
    #dataV =np.array(dataV.values.astype(float))
    #minV =np.min(dataV[:,0])
    #maxV =np.max(dataV[:, 0])
    #i=minV
    #with open('./meanErrorStw.csv', mode='w') as data:
        #data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #data_writer.writerow(['MAE', 'STW'])
        #while i <=maxV:
            #meanErr =np.mean(np.array([k for k in dataV if k[0]>=i and k[0]<=i+0.5])[:,1])
            #data_writer.writerow([meanErr,i])
            #i=i+0.5
    data = pd.read_csv(sFile,delimiter=';')
    data = data.drop(["wind_speed", "wind_dir"],axis=1)
    data = data.values

    k=10000
    trData = data[0:89999]
    k=0
    n=20000
    #subsets=[]
    for i in range(1,5):
        subsetsX.append(trData[k:n*i,0:7])
        subsetsY.append(trData[k:n * i, 7])
        k=n*i+1000

    #indSubsets = []
    #for i in range(0,len(subsets)):
       #X = DANreader.readStatDifferentSubsets(subsets[i],subsets,i)
       #indSubsets.append(X)



    #subsetsX.append(data[:,0:7][0:1000].astype(float))
    #subsetsY.append(data[:, 7][0:1000].astype(float))
    unseenX = data[:, 0:7][90000:].astype(float)
    unseenY = data[:, 7][90000:].astype(float)


    K = range(1,26)
    print("Number of Statistically ind. subsets for training: " + str(len(subsetsX)))

    #K=[10]
    #rangeSubs = k
    stdInU = []
    varTr = []
    models = []
    part = []

    for subsetX, subsetY in zip(subsetsX, subsetsY):

      for modeler in modelers:
        for partitioner in partitioners:
           if partitioner.__class__.__name__=='DelaunayTriPartitioner':
                 partK=np.linspace(0.7,1,4)#[0.5]
                 #np.linspace(0.2,1,11)
                     #[0.6]
           if partitioner.__class__.__name__=='KMeansPartitioner' or partitioner.__class__.__name__=='KMeansPartitionerWS_WA'\
                    or partitioner.__class__.__name__=='KMeansPartitionerWH_WD':
               if modeler.__class__.__name__ == 'PavlosInterpolation':
                 partK = [1]
               if modeler.__class__.__name__=='TriInterpolantModeler' or modeler.__class__.__name__ == 'TensorFlow':
                 partK =K
               else:
                 partK=[25]
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
                    seriesX, targetY, = subsetX,subsetY
                counter=+1

                print("Reading data... Done.")

                # Extract features

                if modeler.__class__.__name__ == 'PavlosInterpolation':
                        k=1
                #partitionsX, partitionsY , partitionLabels=X,Y,W
                #if modeler.__class__.__name__!='TensorFlow':
                # Partition data
                print("Partitioning training set...")
                partitionsX, partitionsY = seriesX , targetY
                NUM_OF_CLUSTERS =k# TODO: Read from command line
                NUM_OF_FOLDS=6
                #if modeler!='TRI':

                if modeler.__class__.__name__ != 'TensorFlowWD':
                        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel  , centroids  = partitioner.clustering(seriesX, targetY, None ,NUM_OF_CLUSTERS, True,k)
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

                if modeler.__class__.__name__!= 'TriInterpolantModeler' and modeler.__class__.__name__!= 'PavlosInterpolation' :
                            #and modeler.__class__.__name__ != 'TensorFlow':
                    modelMap, history,scores, output,genericModel = modeler.createModelsFor(partitionsX, partitionsY, partitionLabels,None,seriesX,targetY)
                            #, genericModel , partitionsXDC
                    #if modeler.__class__.__name__ != 'TensorFlow':
                        #modelMap = dict(zip(partitionLabels, modelMap))
                    print("Creating models per partition... Done")

                    # Get unseen data

                    stdInU.append(np.std(unseenX))
                ##
                print("Reading unseen data... Done")

                # Predict and evaluate on seen data
                if modeler.__class__.__name__  == 'TensorFlowW1':
                    print("Evaluating on seen data...")


                    #_,meanErrorTr, sdErrorTr = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                            #eval.MeanAbsoluteErrorEvaluation(), seriesX,
                            #targetY,
                            #modeler, output, None, None, partitionsX, None)


                    #print ("Mean absolute error on training data: %4.2f (+/- %4.2f standard error)" % (
                        #meanErrorTr, sdErrorTr / sqrt(unseenY.shape[ 0 ])))
                    print("Evaluating on seen data... Done.")

                #Predict and evaluate on unseen data
                print("Evaluating on unseen data...")

                if modeler.__class__.__name__ == 'TensorFlowW1':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                            eval.MeanAbsoluteErrorEvaluation(), unseenX,
                            unseenY,
                            modeler, output, None, None, partitionsX, scores)
                elif modeler.__class__.__name__ == 'TensorFlowW3' or modeler.__class__.__name__ == 'TensorFlowW2':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(
                        eval.MeanAbsoluteErrorEvaluation(), unseenX,
                        unseenY,
                        modeler, output, None, None, partitionsX, scores)
                elif modeler.__class__.__name__=='PavlosInterpolation':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluatePavlosInterpolation(
                        eval.MeanAbsoluteErrorEvaluation(), unseenX,
                        unseenY,
                        modeler, None, None, None, partitionsX, None)


                print ("Mean absolute error on unseen data: %4.2f (+/- %4.2f standard error)"%(meanError, sdError/sqrt(unseenY.shape[0])))
                print("Evaluating on unseen data... Done.")

                print("Standard Deviation of training Data: %4.2f" % (np.std(seriesX)))
                print("Standard Deviation of unseen Data: %4.2f" % (np.std(unseenX)))
                #varExpl=np.sum(np.square(subsetX-unseenX)) / np.square(np.sum(subsetX)-np.sum(unseenX))*100
                #print("Percentage of variance in Training data explained in Unseen Dataset: %4.2f" % (
                        #(varExpl)) + " %")
                # # Evaluate performance
                numOfclusters= len(partitionsX)

                #plotRes.ErrorGraphs.PlotModelConvergence(plotRes.ErrorGraphs(),len(seriesX),len(unseenX),history,numOfclusters,meanErrorTr,sdErrorTr,meanError,sdError)

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
                if modeler.__class__.__name__ == 'TriInterpolantModeler' and numOfclusters==1 or modeler.__class__.__name__ == 'TriInterpolantModeler' \
                        and partitioner.__class__.__name__ == 'DelaunayTriPartitioner':
                    break
                if partitioner.__class__.__name__ == 'DelaunayTriPartitioner' and numOfclusters==1:
                    break


    eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr, errors,models,part)

def initParameters():
    sFile = "./neural_data/marmaras_data.csv"
    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000
    algs=['LI']
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
