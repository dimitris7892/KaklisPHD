from sklearn.model_selection import  train_test_split
import dataReading
import dataReading as dRead
import dataPartitioning as dPart
import dataModeling as dModel
import evaluation as eval
from math import sqrt
import pandas as pd
from pylab import *
#import Danaos_ML_Project.generateProfile as genProf
import mappingData_functions as mpf
import preProcessLegs as preLegs
import pickle


prelegs = preLegs.preProcessLegs(False)
#gener = genProf.BaseProfileGenerator()
mapping = mpf.Mapping()
dread = dataReading.BaseSeriesReader()



def preProcessData(data, datatype):


    stwInd = 12 if datatype == ' raw' else 0
    draftInd = 8 if datatype == ' raw' else 1
    wsInd = 11 if datatype == ' raw' else 3
    focInd = 15 if datatype == ' raw' else 2
    swhInd = 22 if datatype == ' raw' else 5
    wdInd =  10 if datatype == ' raw' else 4
    vslHeadInd = 1 if datatype == ' raw' else 9
    # data =np.array([k for k in data if k[2]=='B' or k[2]=='L'])

    '''wfS = data[:, 11].astype(float) / (1.944)
    wsSen = []
    for i in range(0, len(wfS)):
        wsSen.append(gener.ConvertMSToBeaufort(float(float(wfS[i]))))
    data[:, 11] = wsSen

    # data[:, 11] = wfS

    data[:, 15] = ((data[:, 15]) / 1000) * 1440'''

    for i in range(0, len(data[:, wdInd])):
        if float(data[i, wdInd]) < 0:
            data[i, wdInd] += 360
        data[i, wdInd] = dread.getRelativeDirectionWeatherVessel(float(data[i, vslHeadInd]), float(data[i, wdInd]))

    for i in range(0, len(data[:wdInd])):
        if float(data[i, wdInd]) > 180:
            data[i, wdInd] = float(data[i, wdInd]) - 180  # and  float(k[8])<20

    '''for i in range(0,len(data[:,10])):
        if float(data[i,10]) >=0 and float(data[i,10]) <22.5:
            data[i,10] = 1
        elif float(data[i,10]) >= 22.5 and float(data[i,10]) < 67.5:
            data[i, 10] = 2
        elif float(data[i,10]) >= 67.5 and float(data[i,10]) < 112.5:
            data[i, 10] = 3
        elif float(data[i, 10]) >= 112.5 and float(data[i,10]) < 157.5:
            data[i, 10] = 4
        elif float(data[i,10]) >= 157.5 and float(data[i,10]) < 180.5:
            data[i, 10] = 5'''
    '''for i in range(0, len(data[:,1])):
        if float(data[i,1]) > 180:
            data[i,1] = float(data[i,1]) - 180'''

    '''for i in range(0, len(vslHeadingTD)):
        if float(vslHeadingTD[i]) > 180:
            vslHeadingTD[i] = float(vslHeadingTD[i]) - 180'''

    '''for i in range(0, len(currentDirTD)):
        if float(currentDirTD[i]) > 180:
            currentDirTD[i] = float(currentDirTD[i]) - 180'''

    '''for i in range(0, len(data[:,30])):
        if float(data[i,30]) > 180:
            data[i,30] = float(data[i,30]) - 180'''
    ##################################################
    trData = np.array(
        np.append(data[:, draftInd].reshape(-1, 1), np.asmatrix([data[:, wdInd], data[:, wsInd], data[:, stwInd], data[:, swhInd],
                                                           data[:, focInd]]).T, axis=1)).astype(float)  # data[:,26],data[:,27]
    # trData = np.nan_to_num(trData)
    # trData = np.array(np.append(data[:, 0].reshape(-1, 1),

    # np.asmatrix([data[:, 1], data[:, 3], data[:, 4], data[:, 5]]).T,
    # axis=1)).astype(float)
    # np.array(np.append(data[:,0].reshape(-1,1),np.asmatrix([data[:,1],data[:,2],data[:,3],data[:,4],data[:,7],data[:,8],data[:,9]]).T,axis=1)).astype(float)
    # np.array(np.append(data[:,8].reshape(-1,1),np.asmatrix([data[:,10],data[:,11],data[:,12],data[:,20],data[:,21],data[:,15]]).T,axis=1)).astype(float)

    '''meanFoc = np.mean(trData[:,5 ], axis=0)
    stdFoc = np.std(trData[:, 5], axis=0)
    trData = np.array(
        [k for k in trData if (k[5] >= (meanFoc - (3 * stdFoc))) and (k[5] <= (meanFoc + (3 * stdFoc)))])

    trData = np.array([k for k in trData if  str(k[0])!='nan' and  float(k[2])>=0 and float(k[4])>=0 and (float(k[3])>=9 ) and float(k[5])>0  ]).astype(float)'''
    #trData = np.array([k for k in trData if (float(k[3]) >= 9) ]).astype(float)
    # trData = np.nan_to_num(trData)

    '''genprf = generateProfile.BaseProfileGenerator()
    trDataPorts, trDataNoInPorts = genprf.findCloseToLandDataPoints(trData)
    trData = trDataNoInPorts

    trData = np.array(np.append(trData[:, 0:5], np.asmatrix(trData[:, 7]).T, axis=1))'''
    #######################################################################################
    # np.concatenate([trDataNoInPorts,trDataPorts])
    # plRes.PLotDists(trData)
    e = 0
    # X_train, X_test, y_train, y_test = train_test_split(trData[:, 0:5], trData[:, 5:], test_size=0.26, random_state=42)
    # trData1 =  trData[27000:86000, :]
    # trData2 =  trData[86000:145115,:] #HAMBURG - MUMBAI - #HAMBURG

    # for i in range(0, len(trData)):
    # trData[i] = np.mean(trData[i:i + 15], axis=0)

    '''wd = np.array([k for k in trData])[:, 1]
    for i in range(0, len(wd)):
        if float(wd[i]) > 180:
            wd[i] = float(wd[i]) - 180  # and  float(k[8])<20

    trData[:, 1] = wd'''

    '''wf = np.array([k for k in trData])[:, 2]
    for i in range(0, len(wf)):
        wf[i] = gener.ConvertMSToBeaufort(float(float(wf[i])))
    trData[:, 2] = wf'''

    # trData = trData[:40000]

    for i in range(0, len(trData)):
        trData[i] = np.mean(trData[i:i + 15], axis=0)

    return trData


def main(vessel, algs, cls, fromFile, processedData):

    end, endU, history, future, sFile, start, startU , algs , cls = initParameters(algs, cls)
    # Load data
    if len(sys.argv) >3:
        algs = algs.split(',')
        cls=cls.split(',')
        print(sFile , algs , cls)
        end = int(end)
        endU = int(endU)
        history = int(history)
        future = int(future)
        start = int(start)
        startU = int(startU)


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
        if cl == 'NNCL': partitioners.append(dPart.TensorFlowCl())

    print(modelers)
    ###########################################################################


    # 2nd place BALLAST FLAG
    # 8th place DRAFT
    # 10th place rel WD
    # 11th place WF
    # 12th place SPEED
    # 15th place ME FOC 24H
    # 16th place ME FOC 24H TLGS
    # 17th place TRIM
    # 19th place SteamHours
    # 18th place STW_TLG
    # 20st place swellSWH
    # 21st place  rel swellSW Dir
    # 26th place  lat
    # 27th place  lon
    # 28th place  wsWS

    # 0 draft
    # 1 wd
    # 2 wf
    # 3 stw
    # 4 trim
    # 5 swh
    # 6 swd
    # 7 foc (MT/day)



    '''DANreader = dRead.BaseSeriesReader()
    DANreader.GenericParserForDataExtraction('LEMAG', 'DANAOS', vessel, driver='ORACLE',
                                             server='10.2.5.80',
                                             sid='OR12', usr='shipping', password='shipping',
                                             rawData=True, telegrams=True, companyTelegrams=False,
                                             pathOfRawData='/home/dimitris/Desktop/SEEAMAG')'''

    '''DANreader.GenericParserForDataExtraction('LEMAG', 'DANAOS', 'LEO C', driver='ORACLE',
                                             server='10.2.5.80',
                                             sid='OR12', usr='shipping', password='shipping',
                                             rawData=False, telegrams=True, companyTelegrams=False,
                                             pathOfRawData='/home/dimitris/Desktop/SEEAMAG')'''

    '''DANreader.GenericParserForDataExtraction('LEMAG', 'OCEAN_GOLD', 'PERSEFONE', driver='ORACLE',
                                             server='10.2.5.80',
                                             sid='OR11', usr='oceangold', password='oceangold',
                                             rawData=False, telegrams=True, companyTelegrams=False,
                                             pathOfRawData='/home/dimitris/Desktop/SEEAMAG')'''

    #return



    data = pd.read_csv(sFile,sep=',').values



    ###################PENELOPE##################PENELOPE

    #stwDEITECH, vslHeadingTD , currentSpeedTD, currentDirTD = calculate_stw_TEIDETECH(data,129)
    #data =data[:1000,:]
    vslDir = data[:, 1]
    #currentDir = data[:, 30]
    #currentSpeedKnots = data[:, 29]  # * 1.943
    #sovg = data[:, 31]
    #stwCurr, vslHeading = calculateSTW(sovg, vslDir , currentSpeedKnots, currentDir)
    #stwCurr = np.array(stwCurr)


    #trData = preProcessData(data, 'legs')
    #correctWS_Beaufort("EXPRESS ATHENS")
    #return

    #trData = prelegs.extractDataFromLegs(data, "EXPRESS ATHENS")
    #return
    #trData = prelegs.concatLegs_PrepareForTR( vessel, './correctedLegsForTR/'+vessel+'/')
    #with open('./attentionWeights.pkl', 'rb') as f:
        #attWeights = pickle.load(f)
    attWeights = None

    print(vessel)

    prelegs.extractLegsFromRawCleanedData(vessel, 'raw')

    trData = prelegs.returnCleanedDatasetForTR(vessel, './consProfileJSON_Neural/cleaned_raw_'+vessel+'.csv', 'raw', 11, fromFile, processedData)

    expansion = False

    X_train, X_test, y_train, y_test = train_test_split(trData[:, 0:5], trData[:, 5], test_size=0.1, random_state=42)


    #mapping.writeTrainTestData('DANAOS','HYUNDAI SMART',X_test,y_test,X_train,y_train)
    #return

    #mapping.extractJSON_TestData(9484948,'EXPRESS ATHENS',X_test, y_test,)
    #return
    subsetsX = []
    subsetsY = []


    subsetsX.append(X_train.astype(float))
    subsetsY.append(y_train.astype(float))

    unseenX = X_test.astype(float)
    unseenY = y_test.astype(float)


    print(str(len(X_train)))
    print(str(len(unseenX)))

    ##################################################

    print("Number of Statistically ind. subsets for training: " + str(len(subsetsX)))

    stdInU = []
    varTr = []
    models = []
    K = range(1,25)
    part = []
    subsetInd = 0
    errors = []
    trErrors = []
    var = []
    clusters = []
    cutoff = np.linspace(0.1, 1, 111)


    for subsetX, subsetY in zip(subsetsX, subsetsY):
      for modeler in modelers:
        for partitioner in partitioners:
           if partitioner.__class__.__name__=='DelaunayTriPartitioner':
                 partK=np.linspace(0.7,1,4)
           if partitioner.__class__.__name__=='KMeansPartitioner' or partitioner.__class__.__name__=='KMeansPartitionerWS_WA'\
                    or partitioner.__class__.__name__=='KMeansPartitionerWH_WD':
               if modeler.__class__.__name__ == 'PavlosInterpolation':
                 partK = [1]
               elif modeler.__class__.__name__=='TriInterpolantModeler' or modeler.__class__.__name__ == 'TensorFlow':
                 partK =[1]
               else:
                 partK=[1]
           else:
               partK=[1]
           error = {"errors": []}

           for k in partK:
                if modeler.__class__.__name__ == 'TensorFlowW' and K==1: continue


                print(modeler.__class__.__name__)
                print("Reading data...")
                reader = dRead.BaseSeriesReader()

                ####################################LAROS DATA STATISTICAL TESTS

                print("Reading data... Done.")

                # Extract features


                print("Partitioning training set...")

                seriesX = subsetX
                targetY = subsetY

                NUM_OF_CLUSTERS =k# TODO: Read from command line

                if modeler.__class__.__name__ != 'TensorFlowWD':
                        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel  , centroids  = partitioner.clustering(seriesX, targetY, None, NUM_OF_CLUSTERS, True, k)
                else:
                       #partitionLabels=23

                        partitionsX, partitionsY, partitionLabels, partitionRepresentatives, partitioningModel, tri  = partitioner.clustering(

                            seriesX, targetY, None, NUM_OF_CLUSTERS, True, k)


                print("Partitioning training set... Done.")
                # For each partition create model



                print("Creating models per partition...")


                if modeler.__class__.__name__!= 'TriInterpolantModeler' and modeler.__class__.__name__!= 'PavlosInterpolation' :

                    modelMap, history, scores, output, genericModel = modeler.createModelsFor(partitionsX, partitionsY, partitionLabels

                                                                                            ,None, seriesX, targetY, expansion, vessel, attWeights)

                    print("Creating models per partition... Done")

                    # Get unseen data

                    stdInU.append(np.std(unseenX))
                ##
                print("Reading unseen data... Done")

                # Predict and evaluate on seen data
                print("Evaluating on seen data...")
                '''if modeler.__class__.__name__ == 'TensorFlowW1':
                    _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                        eval.MeanAbsoluteErrorEvaluation(), seriesX,
                        targetY,
                        modeler, output, None, None, partitionsX, scores,subsetInd,'train')
                elif modeler.__class__.__name__ == 'TensorFlowW3' or modeler.__class__.__name__ == 'TensorFlowW2':
                    _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN(
                        eval.MeanAbsoluteErrorEvaluation(), seriesX,
                        targetY,
                        modeler, output, None, None, partitionsX, scores,subsetInd,'train')
                elif modeler.__class__.__name__ == 'PavlosInterpolation':
                    _, meanErrorTr, sdError = eval.MeanAbsoluteErrorEvaluation.evaluatePavlosInterpolation(
                        eval.MeanAbsoluteErrorEvaluation(), subsetX,
                        subsetY,
                        modeler, None, None, None, partitionsX, None, subsetInd,'train')

                print("Mean absolute error on seen data: %4.2f (+/- %4.2f standard error)" % (
                meanErrorTr, sdError / sqrt(unseenY.shape[0])))'''
                print("Evaluating on seen data... Done.")

                #Predict and evaluate on unseen data
                print("Evaluating on unseen data...")


                if modeler.__class__.__name__ == 'TensorFlowW1':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNN1(
                            eval.MeanAbsoluteErrorEvaluation(), unseenX,
                            unseenY,
                            modeler, output, None, None, partitionsX, scores,subsetInd,'test', vessel, False)
                elif modeler.__class__.__name__ == 'TensorFlowW' or modeler.__class__.__name__ == 'TensorFlowW2':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluateKerasNNAvg(
                        eval.MeanAbsoluteErrorEvaluation(), unseenX,
                        unseenY,
                        modeler, output, None, None, partitionsX, genericModel,subsetInd,'test')
                elif modeler.__class__.__name__=='PavlosInterpolation':
                    _, meanError, sdError = eval.MeanAbsoluteErrorEvaluation.evaluatePavlosInterpolation(
                        eval.MeanAbsoluteErrorEvaluation(), unseenX,
                        unseenY,
                        modeler, None, None, None, partitionsX, None,subsetInd,'test')



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

                errors.append(meanError)
                trErrors.append(meanError)

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
      subsetInd=subsetInd+1


    eval.MeanAbsoluteErrorEvaluation.ANOVAtest(eval.MeanAbsoluteErrorEvaluation(), clusters, varTr, trErrors,errors,models,part)

def initParameters(algs, cls):


    sFile = './data/DANAOS/MELISANDE/mappedData.csv'
    #sFile = './consProfileJSON_Neural/cleaned_EXPRESS ATHENS.csv'
    #sFile = './consProfileJSON_Neural/cleaned_legs_EXPRESS ATHENS.csv'

    #prelegs.concatLegs_PrepareForTR( './correctedLegsForTR/EXPRESS ATHENS/')

    # Get file name
    history = 20
    future=30
    start = 10000
    end = 17000
    startU = 30000
    endU = 31000
    algs=algs

    cls=cls


    if len(sys.argv) > 3:
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
