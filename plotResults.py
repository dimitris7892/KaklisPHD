import matplotlib.pyplot as plt
import matplotlib as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pylab import figure,axes
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
import latex
import pandas as pd
import csv

plt.rcParams.update({
    "text.usetex": True,
})

class ErrorGraphs:

    def ErrorGraphswithKandTrlen(self,errors,K,trSize,show,modeler):

        plt.subplot(2, 1, 1)
        plt.plot(K,errors, 'k-')
        plt.title('Error convergence with '+str(modeler))
        plt.xlabel('# of clusters')
        plt.ylabel('Mean Absolute Error')

        plt.subplot(2, 1, 2)
        plt.plot(trSize, errors, 'k-')
        plt.xlabel(' Training Set size(instances)')
        plt.ylabel('Mean Absolute Error')

        if show : plt.show()
        x=1

    def ErrorGraphsForPartioners(self):

        errors= {'data':[]}
        data = pd.read_csv('/home/dimitris/Desktop/models_perf.csv', delimiter=',')
        for row in data.values:

                    if row[0]=='Trerror':continue
                    error={}
                    error['model']=row[3]
                    error['error']=row[2]
                    error['cluster']=row[1]
                    errors['data'].append(error)

        models=['TensorFlowW1','SplineRegressionModeler','LinearRegressionModeler','RandomForestModeler','TensorFlowW',
                'TensorFlowCA']
        with open('/home/dimitris/Desktop/meanErrorThroughClusters.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['models', 'clusterrs', 'error'])
            for m in models:
                for cl in range(1,21):

                    meanError= np.mean([m['error'] for m in [k for k in errors['data'] if k['model']==m and k['cluster']==cl]])
                    data_writer.writerow([m,cl, meanError])




        plt.plot(K, errors, 'k-',label=str(trSize)+ ' Instances')
        plt.legend(loc='upper right', shadow=True)
        plt.title('Error convergence with ' + str(modeler)+" and " + str(clustering))
        if clustering=='KMeansPartitioner':
            plt.xlabel('# of clusters')
        else:
            plt.xlabel('cut-off value')
        plt.ylabel('Mean Absolute Error')

        if show: plt.show()
        x = 1

    def ErrorGraphsWithK(self, errors, var,stdU,K, show, modeler,clustering, trSize=2880):
        Serr=[]
        Lerr=[]
        NNerr=[]

        for v in errors.values()[0]:
            if v["model"]=="SplineRegressionModeler":
               Serr.append(v["error"])
                #Svar.append(v["var"])
            elif v["model"]=="LinearRegressionModeler":
               Lerr.append(v["error"])
                #Lvar.append(v["var"])
            #else:
                #if v["k"] == 1:NNerr.append(v["error"])


        plt.bar(K, Serr, width=0.1,
                 alpha=0.5,
                 color='b',
                 label='Splines')
        plt.bar([ k+0.1 for k in K], Lerr, width=0.1,
                 alpha=0.5,
                 color='g',
                 label='LR')
        #plt.bar([ k + 0.2 for k in K ], NNerr, width=0.1,
        #        alpha=0.5,
        #        color='r',
        #        label='NN')

        plt.title('Error convergence of stat. ind. subsets  with ' + str(clustering) + " through # of Clusters.Var explained between tr. sets "+str(round(var,2))+" %")
        plt.xlabel('# of clusters')
        plt.ylabel('Mean Absolute Error')
        plt.xticks([round(k+0.1,2) for k in K], [k for k in K])
        plt.legend(loc='upper right', shadow=True)
        plt.legend()

        if show: plt.show()
        x = 1


    def ErrorGraphsWithVariance(self, errors, var, show, modeler,clustering, trSize=2880):
        Serr1=[]
        Lerr1=[]
        NNerr=[]
        Serr2 = [ ]
        Lerr2 = [ ]

        for v in errors.values()[0]:
            if v["model"]=="SplineRegressionModeler":
               if v["k"]==1 :Serr1.append(v["error"])
               if v[ "k" ] == 2: Serr2.append(v[ "error" ])
                #Svar.append(v["var"])
            elif v["model"]=="LinearRegressionModeler":
                if v[ "k" ] == 1:Lerr1.append(v["error"])
                if v[ "k" ] == 2: Lerr2.append(v[ "error" ])
                #Lvar.append(v["var"])
            else:
               NNerr.append(v["error"])

        #plt.subplot(2, 1, 1)
        plt.bar(var, Serr2, width=0.01,
                 alpha=0.5,
                 color='b',
                 label='Splines')
        plt.bar([ v+0.1 for v in var], Lerr2, width=0.01,
                 alpha=0.5,
                 color='g',
                 label='LR')
        plt.bar([ v + 0.2 for v in var ], NNerr, width=0.01,
                alpha=0.5,
                color='r',
                label='NN')

        plt.title('Error convergence of stat. ind. subsets  with ' + str(clustering) +" k=2 "+ " through Variance in Tr. Set")
        plt.xlabel('Variance')
        plt.ylabel('Mean Absolute Error')
        plt.xticks([round(v+0.1,2) for v in var], [round(v,2) for v in var])
        plt.legend(loc='upper right', shadow=True)
        plt.legend()
        ########################subplot

        if show: plt.show()
        x = 1

    def ThreeDErrorGraphwithKandTrlen(self,errors,K,trSize,show,):
         ax = plt.axes(projection='3d')

         # Data for a three-dimensional line
         zline = np.array(errors)
         xline = np.array(K)
         yline = np.array(trSize)
         ax.set_xlabel('# of CLusters')
         ax.set_ylabel('Training Set size(instances)')
         ax.set_zlabel('MAE %')
         ax.set_zlim(0, np.max(errors))
         ax.plot3D(xline, yline, zline, 'red')
         if show :plt.show()
         x=1

    def PlotTrueVsPredLine(self,):
        from pylab import figure

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}

        plt.rc('font', **font)

        dataErrorxiNN = pd.read_csv('./testError/TESTerrorPercRPMxiNN1_0.csv', delimiter=',', skiprows=1)
        #dataErrorxiLSTM = pd.read_csv('/home/dimitris/Desktop/resultsNEW/TESTerrorPercRPMxiLSTM1_0.csv', delimiter=',', skiprows=1)
        dataErrorxiLSTM = pd.read_csv('./testError/TESTerrorPercRPMxiLSTM1_0.csv', delimiter=',',skiprows=1)
        dataErrorAM = pd.read_csv('./testError/TESTerrorPercRPM_AM_0.csv', delimiter=',',skiprows=1)

        dataErrxiNN = dataErrorxiNN.values
        #dataErrorLSTM = dataErrorLSTM.values
        dataErrorAM = dataErrorAM.values
        dataErrorxiLSTM = dataErrorxiLSTM.values

        errMeanxiNN = []
        errMeanLSTM = []
        errMeanxiLSTM = []
        errMeanAM=[]
        rpmMean = []
        i = 0
        xAxisSize =int(len(dataErrorxiLSTM)/12)
        while i < xAxisSize:
            #errMeanxiNN.append(np.mean(dataErrxiNN[i:i + 1, 1]))
            #errMeanLSTM.append(np.mean(dataErrorLSTM[i:i + 1, 1]))
            #errMeanxiLSTM.append(np.mean(dataErrorxiLSTM[i:i + 1, 1]))

            #rpmMean.append(np.mean(dataErrxiNN[i:i + 1, 0]))

            errMeanxiNN.append(dataErrxiNN[i, 1])
            errMeanAM.append(dataErrorAM[i, 1])
            errMeanxiLSTM.append(dataErrorxiLSTM[i, 1])

            rpmMean.append(dataErrorxiLSTM[i, 0])


            i = i + 1

        '''errMean = []
        errMeanMA = []
        focMean = []
        mae = []
        stwMean=[]
        draftMean=[]
        wsMean=[]
        #meanPercError = []
        i = 0
        while i < len(dataErr):
            errMean.append(np.mean(dataErr[i:i + 20, 0]))
            #errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
            #focMean.append(np.mean(dataErr[i:i + 20, 1]))
            meanPercError= dataErr[i, 2]
            if meanPercError > 50:
                stwMean.append(dataErr[i, 3])
                draftMean.append(dataErr[i, 4])
                wsMean.append(dataErr[i, 6])

            #stwMean.append(np.mean(dataErr[i:i + 20, 3]))
            #draftMean.append(np.mean(dataErr[i:i + 20, 4] ))
            #wsMean.append(np.mean(dataErr[i:i + 20, 6] ))
            mae.append(abs(np.mean(dataErr[i:i + 20, 1]) - np.mean(dataErr[i:i + 20, 0])))
            i = i + 1
            #i = i + 20

        errMean = np.array(errMean)
        errMeanMA = np.array(errMeanMA)
        focMean = np.array(focMean)

        features=[]
        values=[]
        groups=[]

        barWidth = 0.25
        r1 = np.arange(len(stwMean))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]

        # Make the plot
        plt.bar(r1, stwMean, color='#7f6d5f', width=barWidth, edgecolor='white', label='stw')
        plt.bar(r2, wsMean, color='#557f2d', width=barWidth, edgecolor='white', label='ws')
        plt.bar(r3, draftMean, color='#2d7f5e', width=barWidth, edgecolor='white', label='draft')

        # Add xticks on the middle of the group bars
        for i in range(0,len(stwMean)):
            groups.append( str(i))
        plt.xlabel('group', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(stwMean))], groups)

        #for i in range(0,len(stwMean)):
            #features.append(['stw'+str(i), 'draft'+str(i), 'ws'+str(i)])
            #values.append([int(stwMean[i]),int(draftMean[i]),int(wsMean[i])])
            #values = [stwMean, draftMean, wsMean]
            #plt.bar(features, values)


        # Create legend & Show graphic
        plt.legend()
        plt.figure(figsize=(3, 4))
        plt.show()'''
        fig = figure()


        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), errMeanxiNN, '-', c='green', label=r'\textbf{RPM Predictions with ExtendedSpace Network, acc: 97.05\% }')
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=stwMean,c='red', alpha=0.5)
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=draftMean, c='blue', alpha=0.5)

        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), errMeanxiLSTM, '-', c='blue', label=r'\textbf{RPM Predictions with SplineLSTM Network, acc: 97.93\%}')
        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), errMeanAM, '-', c='yellow', label=r'\textbf{RPM Predictions with Analytical Method, acc: 97.86\%}')
        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), rpmMean, '-', c='red', label=r'\textbf{Actual RPM}')
        #plt.fill_between(np.linspace(0, len(rpmMean), len(rpmMean)), abs(np.array(rpmMean) - 1.434),abs(np.array(rpmMean) + 1.434),color='gray', alpha=0.2)

        plt.ylabel(r'\textbf{RPM}',fontsize=19)
        plt.xlabel(r'\textbf{Observations}',fontsize=19)
        #plt.title('Performance comparison of top 3 methods')
        plt.legend()
        plt.grid()

        fig.set_size_inches([18.375, 10.375])
        plt.savefig('/home/dimitris/Desktop/resultsNEW/top3compare.eps', format='eps')
        plt.show()

        x=0

    def PlotExpRes(self,dataErrorFoc,alg):

        fig = plt.figure()

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 40}

        plt.rc('font', **font)
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5

        linestyle_tuple = [
            ('loosely dotted', (0, (1, 10))),
            ('dotted', (0, (1, 1))),
            ('densely dotted', (0, (1, 1))),

            ('loosely dashed', (0, (5, 10))),
            ('dashed', (0, (5, 5))),
            ('densely dashed', (0, (5, 1))),

            ('loosely dashdotted', (0, (3, 10, 1, 10))),
            ('dashdotted', (0, (3, 5, 1, 5))),
            ('densely dashdotted', (0, (3, 1, 1, 1))),

            ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

        results={}
        dataset = 0
        results['EstimatorResults'] = []
        #outerItem = {"estimator": , "speed": (velMin + velMax) / 2, "cells": []}
        #dataRes = pd.read_csv('/home/dimitris/Desktop/LAROS/NEWres_1.csv').values
        '''dataRes = pd.read_csv('/home/dimitris/Desktop/resultsNEW/kmeansNEW.csv', delimiter=',').values
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
        Errors = []
        Clusters = []

        for i in range(0,5):
              listOfDict = [k for k in results['EstimatorResults'] if k['dataset'] == i  and k['partitioner']=='KMeansPartitioner']
              listOfErrors = [x['error'] for x in listOfDict]
              listOfClusters = [x['cluster'] for x in listOfDict]
              minIndex = listOfErrors.index(min([x['error'] for x in listOfDict]))
              minErrors.append(listOfErrors[minIndex])
              minClusters.append(listOfClusters[minIndex])
              Errors.append(listOfErrors)
              Clusters.append(listOfClusters)
              ####
        from scipy.stats import spearmanr
        meanListOfClusters = np.mean(Clusters,axis=0)
        meanListOfErrors = np.mean(Errors, axis=0)
        print(spearmanr(meanListOfClusters,meanListOfErrors))'''
        #dataErrorFoc = pd.read_csv('/home/dimitris/Desktop/___RES/DTC.csv', delimiter=',')
        #dataErrorFoc = pd.read_csv('/home/dimitris/Desktop/resultsNEW/kmeansNEW.csv', delimiter=',')

        errSR = []
        errWINN = []
        errBSPNN = []
        errXINN = []
        errRF = []
        errLR = []

        #while i <  len(dataErrorFoc):

        errSR=(dataErrorFoc.loc[dataErrorFoc['models'] == 'SR']['error'].values)
        errRF=(dataErrorFoc.loc[dataErrorFoc['models'] == 'RF']['error'].values)
        errLR=(dataErrorFoc.loc[dataErrorFoc['models'] == 'LR']['error'].values)
        errWINN=(dataErrorFoc.loc[dataErrorFoc['models'] == 'wavg-wiBSpNN']['error'].values)
        errBSPNN=(dataErrorFoc.loc[dataErrorFoc['models'] == 'wavg-BSpNN']['error'].values)
        errXINN=(dataErrorFoc.loc[dataErrorFoc['models'] == 'wavg-xiBSpNN']['error'].values)



        #np.linspace(1, 10, len(errSR))
        #[1, 2, 3, 4, 5, 8]
        xAxis = np.linspace(1, 10, len(errSR))

        '''plt.plot(xAxis, errSR, '-o', c='green', label='SR')
        plt.plot(xAxis, errRF, '-o', c='red',
                 label='RF')
        plt.plot(xAxis, errLR, '-o', c='blue',
                 label='LR')
        plt.plot(xAxis, errWINN, '-o', c='orange',
                 label='wavg-SplineWeightInitNN')
        plt.plot(xAxis, errBSPNN, '-o', c='yellow',
                 label='wavg-BSpliNNet')'''

        plt.plot(xAxis, errSR,  '-o' ,c='black', label='SR',linewidth=2)
        plt.plot(xAxis, errRF,linestyle= linestyle_tuple[3][1],c='black',
                 label='RF',linewidth=2)
        plt.plot(xAxis, errLR, linestyle= 'dotted',c='black',
                 label='LR',linewidth=2)
        plt.plot(xAxis, errWINN, linestyle= 'dashdot',c='black',
                 label='wavg-SplineWeightInitNN',linewidth=2)
        plt.plot(xAxis, errBSPNN, linestyle= linestyle_tuple[9][1],c='black',
                 label='wavg-BSpliNNet',linewidth=2)
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=stwMean,c='red', alpha=0.5)
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=draftMean, c='blue', alpha=0.5)

        plt.plot(xAxis, errXINN, linestyle='solid',c='black', label='wavg-ExtendedSpace',linewidth=2)
        #plt.plot(np.linspace(0, 197, 197), focMean, '-', c='red', label='Actual RPM')
        #plt.fill_between(np.linspace(0, 100, 100), errMean - mae, errMean + mae,color='gray', alpha=0.2)
        plt.ylabel('MAE',)
        plt.xlabel(r'\textbf{\# of clusters}',)
        #plt.title('Model convergence using K-Means clustering')
        legend_properties = {'weight': 'bold','size':17}
        plt.legend(prop=legend_properties)
        plt.grid()
        #plt.rc('axes', labelsize=41)
        #plt.show()
        fig.set_size_inches([17.375, 10])
        plt.savefig('/home/dimitris/Desktop/'+alg+'_clusters.eps', format='eps')
        x=0

    def boxPLots(self):
        # function for setting the colors of the box plots pairs
        def setBoxColors(bp):
            setp(bp['boxes'][0], color='blue')
            setp(bp['caps'][0], color='blue')
            setp(bp['caps'][1], color='blue')
            setp(bp['whiskers'][0], color='blue')
            setp(bp['whiskers'][1], color='blue')
            setp(bp['fliers'][0], color='blue')
            setp(bp['fliers'][1], color='blue')
            setp(bp['medians'][0], color='blue')

            setp(bp['boxes'][1], color='red')
            setp(bp['caps'][2], color='red')
            setp(bp['caps'][3], color='red')
            setp(bp['whiskers'][2], color='red')
            setp(bp['whiskers'][3], color='red')
            #setp(bp['fliers'][2], color='red')
            #setp(bp['fliers'][3], color='red')
            setp(bp['medians'][1], color='red')

        # Some fake data to plot
        font = {'family': 'normal',
                'weight': 'normal',
                'size': 23}

        plt.rc('font', **font)

        SR = [[3.1, 2.8, 2.9, 1.9 ], [1.76, 1.77,1.89,1.97]]
        LR = [[3.1, 2.8, 2.9, 2.1 ], [2.26, 2.45,2.39,2.57]]
        RF = [[2.2, 1.9, 2.2, 2.76], [2.76, 3.67, 3.59, 2.97]]

        wiBSPNN = [[2.9, 2.2, 2.2, 2.7], [2.1, 2.17, 1.45, 2.37]]
        xiBSPNN = [[3.21, 2.3, 2.8, 3.4], [1.36, 1.48, 2.59, 1.97]]
        BSPNN = [[3.23, 3.8, 2.9, 3.7], [2.53, 3.1, 2.59, 3.17]]


        xiLSTMNN = [[1.67,2.41,2.13,1.45,1.3],[]]
        LSTMNN = [[3.51,4.45,4.24,3.65,4.29],[]]

        AN =[[1.28309551,1.97549886,1.56220609,1.11775517,1.53046567],[]]

        errSR = []
        errWINN = []
        errBSPNN = []
        errXINN = []
        errRF = []
        errLR = []

        # while i <  len(dataErrorFoc):

        '''errSR = (dataErrorDT.loc[dataErrorDT['models'] == 'SR']['error'].values)
        errSR1 =errSR[0]
        errSRN = min(errSR[1:len(errSR)])

        errRF = (dataErrorDT.loc[dataErrorDT['models'] == 'RF']['error'].values)
        errLR = (dataErrorDT.loc[dataErrorDT['models'] == 'LR']['error'].values)
        errWINN = (dataErrorDT.loc[dataErrorDT['models'] == 'wavg-wiBSpNN']['error'].values)
        errBSPNN = (dataErrorDT.loc[dataErrorDT['models'] == 'wavg-BSpNN']['error'].values)
        errXINN = (dataErrorDT.loc[dataErrorDT['models'] == 'wavg-xiBSpNN']['error'].values)'''

        fig = figure()
        ax = axes()
        #hold(True)

        # first boxplot pair
        bp = boxplot(AN, positions=[1, 2], widths=0.6)
        setBoxColors(bp)

        # first boxplot pair
        bp = boxplot(SR, positions=[6, 7], widths=0.6)
        setBoxColors(bp)

        # second boxplot pair
        bp = boxplot(LR, positions=[11, 12], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(RF, positions=[16, 17], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(BSPNN, positions=[24, 25], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(wiBSPNN, positions=[41, 42], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(xiBSPNN, positions=[60, 61], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(xiLSTMNN, positions=[74.5, 75], widths=0.6)
        setBoxColors(bp)

        bp = boxplot(LSTMNN, positions=[87.5, 88], widths=0.6)
        setBoxColors(bp)

        # set axes limits and labels
        xlim(0, 90)
        ylim(0, 5)
        ax.set_ylabel('$MAE$',fontsize=25)
        ax.set_xticklabels([r'\textbf{AN}',r'\textbf{SR}', r'\textbf{LR}', r'\textbf{RF}',r'\textbf{B-SpliNNet}',r'\textbf{SplineWeightInitNN}',r'\textbf{ExtendedSpace}',r'\textbf{SplineLSTM}',r'\textbf{BaseLSTM}'])
        ax.set_xticks([1.5, 6.5, 11.5, 16.5, 24.5, 41.5,60,74.5,87.5])

        #ax.set_ylabel('MAE', )

        # draw temporary red and blue lines and use them to create a legend
        hB, = plot([1, 1], 'b-')
        hR, = plot([1, 1], 'r-')
        legend((hB, hR), (r'\textbf{Without Clustering}', r'\textbf{With Clustering}'))
        hB.set_visible(False)
        hR.set_visible(False)

        fig.set_size_inches([17.375, 8.375])
        plt.savefig('/home/dimitris/Desktop/resultsNEW/boxcompare.eps', format='eps')
        show()
        x=0

    def boxPLotsKMDT(self):
        # function for setting the colors of the box plots pairs
        def setBoxColors(bp):
            setp(bp['boxes'][0], color='blue')
            setp(bp['caps'][0], color='blue')
            setp(bp['caps'][1], color='blue')
            setp(bp['whiskers'][0], color='blue')
            setp(bp['whiskers'][1], color='blue')
            setp(bp['fliers'][0], color='blue')
            setp(bp['fliers'][1], color='blue')
            setp(bp['medians'][0], color='blue')

            setp(bp['boxes'][1], color='red')
            setp(bp['caps'][2], color='red')
            setp(bp['caps'][3], color='red')
            setp(bp['whiskers'][2], color='red')
            setp(bp['whiskers'][3], color='red')
            # setp(bp['fliers'][2], color='red')
            # setp(bp['fliers'][3], color='red')
            setp(bp['medians'][1], color='red')

        # Some fake data to plot
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 25}

        plt.rc('font', **font)



        dataErrorDT = pd.read_csv('/home/dimitris/Desktop/resultsNEW/DTC1.csv', delimiter=',')
        dataErrorKM = pd.read_csv('/home/dimitris/Desktop/resultsNEW/kmeans1.csv', delimiter=',')



        errSRKM=(dataErrorKM.loc[dataErrorKM['models'] == 'SR']['error'].values)
        errRFKM=(dataErrorKM.loc[dataErrorKM['models'] == 'RF']['error'].values)
        errLRKM=(dataErrorKM.loc[dataErrorKM['models'] == 'LR']['error'].values)
        errWINNKM=(dataErrorKM.loc[dataErrorKM['models'] == 'wavg-wiBSpNN']['error'].values)
        errBSPNNKM=(dataErrorKM.loc[dataErrorKM['models'] == 'wavg-BSpNN']['error'].values)
        errXINNKM=(dataErrorKM.loc[dataErrorKM['models'] == 'wavg-xiBSpNN']['error'].values)

        errSRDT = (dataErrorDT.loc[dataErrorDT['models'] == 'SR']['error'].values)
        errRFDT = (dataErrorDT.loc[dataErrorDT['models'] == 'RF']['error'].values)
        errLRDT = (dataErrorDT.loc[dataErrorDT['models'] == 'LR']['error'].values)
        errWINNKDT = (dataErrorDT.loc[dataErrorDT['models'] == 'wavg-wiBSpNN']['error'].values)
        errBSPNNDT = (dataErrorDT.loc[dataErrorDT['models'] == 'wavg-BSpNN']['error'].values)
        errXINNDT = (dataErrorDT.loc[dataErrorDT['models'] == 'wavg-xiBSpNN']['error'].values)

        SR = [errSRKM,errSRDT]
        LR = [errLRKM, errLRDT]
        RF = [errRFKM, errRFDT]

        wiBSPNN = [errWINNKM, errWINNKDT]
        xiBSPNN = [errXINNKM, errXINNDT]
        BSPNN = [errBSPNNKM, errBSPNNDT]

        #xiLSTMNN = [[1.67, 2.41, 2.13, 1.45, 1.3], []]
        #LSTMNN = [[1.51, 2.45, 2.24, 1.65, 1.29], []]

        #AN = [[1.28309551, 1.97549886, 1.56220609, 1.11775517, 1.53046567], []]


        fig = figure()
        ax = axes()
        # hold(True)



        # first boxplot pair
        bp = boxplot(SR, positions=[1, 2], widths=0.6)
        setBoxColors(bp)

        # second boxplot pair
        bp = boxplot(LR, positions=[5, 6], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(RF, positions=[9,10], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(BSPNN, positions=[13, 14], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(wiBSPNN, positions=[20.5, 21.5], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(xiBSPNN, positions=[29, 30], widths=0.6)
        setBoxColors(bp)

        # set axes limits and labels
        xlim(0, 32.5)
        ylim(0, 4.5)
        ax.set_ylabel('$MAE$')
        ax.set_xticklabels(
            [r'$\textbf{SR}$', r'$\textbf{LR}$', r'$\textbf{RF}$', r'$\textbf{B-SpliNNet}$', r'$\textbf{SplineWeightInitNN}$', r'$\textbf{ExtendedSpace}$'])
        ax.set_xticks([1, 5.5, 9.5, 13.5, 21, 29.5])

        # draw temporary red and blue lines and use them to create a legend
        hB, = plot([1, 1], 'b-')
        hR, = plot([1, 1], 'r-')
        legend((hB, hR), (r'\textbf{K-Means}', r'\textbf{DTC}'))
        hB.set_visible(False)
        hR.set_visible(False)

        fig.set_size_inches([17.375, 8.375])
        #plt.savefig('/home/dimitris/Desktop/resultsNEW/boxcompareKMDTC.eps', format='eps')
        show()
        x = 0

    def computeMeansStd(self):

        dataErrorLSTM = pd.read_csv('/home/dimitris/Desktop/resultsNEW/LSTM_xiNNLSTM.csv', delimiter=',')

        x=0
    
    def generateGraphVRPM(self):

        # Some fake data to plot
        font = {'family': 'normal',
                'weight': 'normal',
                'size': 23}

        plt.rc('font', **font)

        data = pd.read_csv('./kaklis.csv')
        data = data.values[0:20000]
        data = np.array([k for k in data[0:, 2:23] if k[3] > 3 and k[5] > 10])  #

        '''for i in range(0, len(data)):
            data[i] = np.mean(data[i:i +15], axis=0)'''

        stw = np.asarray([np.nan_to_num(np.float(x)) for x in data[:, 3]])
        rpm = np.asarray([np.nan_to_num(np.float(y)) for y in data[:, 5]])

        xi = np.array(stw)
        yi = np.array(rpm)

        # Change color with c and alpha
        p2 = np.poly1d(np.polyfit(xi, yi, 2))
        xp = np.linspace(min(xi), max(xi), 100)
        #plt.plot([], [], '.', xp, p2(xp))

        plt.scatter(xi, yi, linewidth=2,c="red", alpha=0.3)
        plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 2))
        plt.xlabel(r"\textbf{Speed (knots)}")
        plt.ylabel(r"\textbf{RPM}")
        plt.grid()
        plt.show()
        minrpm = np.min(rpm)
        maxrpm = np.max(rpm)

        minstw = np.min(stw)
        maxstw = np.max(stw)

        rpmsApp = []
        meanSpeeds = []
        stdSpeeds = []
        ranges = []
        k = 0
        #i = minrpm if minrpm > 0 else 1
        i = minstw if minstw > 0 else 1

        
        rpmsPLot = []
        speedsPlot = []
     
        while i <= maxstw:
            # workbook._sheets[sheet].insert_rows(k+27)
         
            rpmArray = np.array([k for k in data if float(k[3]) >= i and float(k[3]) <= i +1])
            #rpmsApp.append(str(np.round(rpmArray.__len__() / rpmAmount * 100, 2)) + '%')


           
            if rpmArray.__len__() > 5:
                rpmsPLot.append(rpmArray.__len__())
                #speedsPlot.append(np.round((np.min(np.nan_to_num(rpmArray[:, 3].astype(float)))), 2))
                speedsPlot.append(i)
                ranges.append(np.round((np.mean(np.nan_to_num(rpmArray[:, 5].astype(float)))), 2))
            i += 1
            k += 1

        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(rpmsPLot)

        plt.clf()
        # Change color with c and alpha
        p2 = np.poly1d(np.polyfit(xi, yi, 2,w=zi))
        xp = np.linspace(min(xi), max(xi), 100)
        plt.plot([], [], '.', xp, p2(xp))

        plt.scatter(xi, yi, s=zi, c="red", alpha=0.4, linewidth=4)
        plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
        plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
        plt.xlabel("Speed (knots)")
        plt.ylabel("RPM")
        plt.title("Density plot", loc="center")
        fig = plt.gcf()
        fig.set_size_inches(17.5, 9.5)

        dataModel = KMeans(n_clusters=3)
        zi = zi.reshape(-1, 1)
        dataModel.fit(zi)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        ziSorted = np.sort(centroids, axis=0)

        sizeZ =[100,300,500]
        k=0
        for z in ziSorted:
            plt.scatter([], [], c='r', alpha=0.5, s=sizeZ[k],
                        label=str(int(np.floor(z[0]))) + ' obs.')
            k=k+1
        plt.legend(scatterpoints=1, frameon=False, labelspacing=2, title='# of obs')

        plt.show()
        #fig.savefig('/home/dimitris/Desktop/newResults/v_rpm.eps',)
        f=0
