import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pylab import figure,axes
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

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
        import pandas as pd
        import csv
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

    def ThreeDErrorGraphwithKandTrlen(self,errors,K,trSize,show):
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
                'size': 14}

        plt.rc('font', **font)

        dataErrorxiNN = pd.read_csv('/home/dimitris/Desktop/resultsNEW/TESTerrorPercRPMxiNN1_0.csv', delimiter=',', skiprows=1)
        dataErrorxiLSTM = pd.read_csv('/home/dimitris/Desktop/resultsNEW/TESTerrorPercRPMxiLSTM1_0.csv', delimiter=',', skiprows=1)
        dataErrorLSTM = pd.read_csv('/home/dimitris/Desktop/resultsNEW/TESTerrorPercRPMLSTM1_0.csv', delimiter=',', skiprows=1)

        dataErrxiNN = dataErrorxiNN.values
        dataErrorLSTM = dataErrorLSTM.values
        dataErrorxiLSTM = dataErrorxiLSTM.values

        errMeanxiNN = []
        errMeanLSTM = []
        errMeanxiLSTM = []
        rpmMean = []
        i = 0
        xAxisSize =int(len(dataErrxiNN)/12)
        while i < xAxisSize:
            #errMeanxiNN.append(np.mean(dataErrxiNN[i:i + 1, 1]))
            #errMeanLSTM.append(np.mean(dataErrorLSTM[i:i + 1, 1]))
            #errMeanxiLSTM.append(np.mean(dataErrorxiLSTM[i:i + 1, 1]))

            #rpmMean.append(np.mean(dataErrxiNN[i:i + 1, 0]))

            errMeanxiNN.append(dataErrxiNN[i, 1])
            errMeanLSTM.append(dataErrorLSTM[i, 1])
            errMeanxiLSTM.append(dataErrorxiLSTM[i, 1])

            rpmMean.append(dataErrxiNN[i, 0])


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


        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), errMeanxiNN, '-', c='green', label='RPM Predictions with Sequential enriched input Network, acc: 96.8% acc')
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=stwMean,c='red', alpha=0.5)
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=draftMean, c='blue', alpha=0.5)

        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), errMeanxiLSTM, '-', c='blue', label='RPM Predictions with LSTM enriched input Network, acc: 97.7%')
        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), errMeanLSTM, '-', c='yellow', label='RPM Predictions with simple LSTM Network, acc: 97.06%')
        plt.plot(np.linspace(0, len(rpmMean), len(rpmMean)), rpmMean, '-', c='red', label='Actual RPM')
        #plt.fill_between(np.linspace(0, len(rpmMean), len(rpmMean)), abs(np.array(rpmMean) - 1.434),abs(np.array(rpmMean) + 1.434),color='gray', alpha=0.2)

        plt.ylabel('RPM',fontsize=19)
        plt.xlabel('Observations',fontsize=19)
        #plt.title('Performance comparison of top 3 methods')
        plt.legend()
        plt.grid()

        fig.set_size_inches([18.375, 10.375])
        plt.savefig('/home/dimitris/Desktop/resultsNEW/top3compare.eps', format='eps')
        plt.show()

        x=0

    def PlotExpRes(self,):

        fig = plt.figure()
        ax = axes()
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 15}

        plt.rc('font', **font)

        #dataErrorFoc = pd.read_csv('/home/dimitris/Desktop/resultsNEW/DTC.csv', delimiter=',')
        dataErrorFoc = pd.read_csv('/home/dimitris/Desktop/resultsNEW/kmeans.csv', delimiter=',')

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





        plt.plot(np.linspace(1, 10, 10), errSR, '-', c='green', label='SR')
        plt.plot(np.linspace(1, 10, 10), errRF, '-', c='red',
                 label='RF')
        plt.plot(np.linspace(1, 10, 10), errLR, '-', c='blue',
                 label='LR')
        plt.plot(np.linspace(1, 10, 10), errWINN, '-', c='orange',
                 label='wavg-wiBSpNN')
        plt.plot(np.linspace(1, 10, 10), errBSPNN, '-', c='yellow',
                 label='wavg-BSpNN')
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=stwMean,c='red', alpha=0.5)
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=draftMean, c='blue', alpha=0.5)

        plt.plot(np.linspace(1, 10, 10), errXINN, '-', c='black', label='wavg-xiBSpNN')
        #plt.plot(np.linspace(0, 197, 197), focMean, '-', c='red', label='Actual RPM')
        #plt.fill_between(np.linspace(0, 100, 100), errMean - mae, errMean + mae,color='gray', alpha=0.2)
        plt.ylabel('MAE',fontsize=19)
        plt.xlabel('# of clusters',fontsize=19)
        #plt.title('Model convergence using K-Means clustering')
        plt.legend()
        plt.grid()
        #plt.rc('axes', labelsize=41)
        plt.show()
        fig.set_size_inches([17.375, 8.375])
        plt.savefig('/home/dimitris/Desktop/resultsNEW/kmeanserror.eps', format='eps')
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
                'size': 19}

        plt.rc('font', **font)

        SR = [[3.1, 2.8, 2.9, 1.9 ], [1.76, 1.77,1.89,1.97]]
        LR = [[3.1, 2.8, 2.9, 2.1 ], [2.26, 2.45,2.39,2.57]]
        RF = [[2.2, 1.9, 2.2, 2.76], [2.76, 3.67, 3.59, 2.97]]

        wiBSPNN = [[2.9, 2.2, 2.2, 2.7], [2.1, 2.17, 1.45, 2.37]]
        xiBSPNN = [[1.21, 1.3, 1.5, 1.4], [1.36, 1.48, 2.59, 1.97]]
        BSPNN = [[3.23, 3.8, 2.9, 3.7], [2.53, 3.1, 2.59, 3.17]]


        xiLSTMNN = [[1.67,2.41,2.13,1.45,1.3],[]]
        LSTMNN = [[1.51,2.45,2.24,1.65,1.29],[]]

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
        bp = boxplot(SR, positions=[4, 5], widths=0.6)
        setBoxColors(bp)

        # second boxplot pair
        bp = boxplot(LR, positions=[7, 8], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(RF, positions=[10, 11], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(BSPNN, positions=[14, 15], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(wiBSPNN, positions=[21, 22], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(xiBSPNN, positions=[28, 29], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(xiLSTMNN, positions=[35.5, 36], widths=0.6)
        setBoxColors(bp)

        bp = boxplot(LSTMNN, positions=[41.5, 42], widths=0.6)
        setBoxColors(bp)

        # set axes limits and labels
        xlim(0, 43)
        ylim(0, 4)
        ax.set_ylabel('MAE',fontsize=19)
        ax.set_xticklabels(['AN','SR', 'LR', 'RF','wavg-BSpNN','wavg-wiBSpNN','wavg-xiBSpNN','xiBSpLSTMNN','LSTMNN'])
        ax.set_xticks([1, 4.5, 7.5, 10.5, 14.5, 21.5,28.5,35.5,41.5])

        #ax.set_ylabel('MAE', )

        # draw temporary red and blue lines and use them to create a legend
        hB, = plot([1, 1], 'b-')
        hR, = plot([1, 1], 'r-')
        legend((hB, hR), ('Without Clustering', 'With Clustering'))
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
        bp = boxplot(wiBSPNN, positions=[20, 21], widths=0.6)
        setBoxColors(bp)

        # thrid boxplot pair
        bp = boxplot(xiBSPNN, positions=[27, 28], widths=0.6)
        setBoxColors(bp)

        # set axes limits and labels
        xlim(0, 29.5)
        ylim(0, 4.5)
        ax.set_ylabel('MAE')
        ax.set_xticklabels(
            ['SR', 'LR', 'RF', 'wavg-BSpNN', 'wavg-wiBSpNN', 'wavg-xiBSpNN'])
        ax.set_xticks([1, 5.5, 9.5, 13.5, 20.5, 27.5])

        # draw temporary red and blue lines and use them to create a legend
        hB, = plot([1, 1], 'b-')
        hR, = plot([1, 1], 'r-')
        legend((hB, hR), ('K-Means', 'DTC'))
        hB.set_visible(False)
        hR.set_visible(False)

        fig.set_size_inches([17.375, 8.375])
        plt.savefig('/home/dimitris/Desktop/resultsNEW/boxcompareKMDTC.eps', format='eps')
        show()
        x = 0

    def computeMeansStd(self):

        dataErrorLSTM = pd.read_csv('/home/dimitris/Desktop/resultsNEW/LSTM_xiNNLSTM.csv', delimiter=',')

        x=0
