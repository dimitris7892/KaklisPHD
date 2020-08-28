import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sqrt
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
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

    def ErrorGraphsForPartioners(self, errors, K, trSize, show, modeler,clustering):

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

    def PlotModelConvergence(self,lenX,lenUnseenX,history,numOfclusters,meanErrorTr,sdErrorTr,meanError,sdError):

        # Plot training & validation loss values

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        #plt.ylim(0,2)
        plt.title('Model loss with ' + str(numOfclusters) + ' cluster(s)')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        legend = plt.legend(['Train', 'Test'], loc='upper left')
        plt.gca().add_artist(legend)
        legend1=plt.legend(["Mean absolute error on training data ("+str(lenX)+" obs.) : %4.2f (+/- %4.2f standard error)" % (
        meanErrorTr, sdErrorTr / sqrt(lenUnseenX))],  loc='upper right')
        plt.gca().add_artist(legend1)
        legend2 = plt.legend(["Mean absolute error on unseen data ("+str(lenUnseenX)+" obs.) : %4.2f (+/- %4.2f standard error)"%(meanError, sdError/sqrt(lenUnseenX))],
                   loc='center right',bbox_to_anchor=(0,0,1,1))
        plt.gca().add_artist(legend2)
        plt.show()
        x=0

    def PlotTrueVsPredLine(self,):

        dataErrorFoc = pd.read_csv('C:/Users/dkaklis/Desktop/TESTerrorPercFOC7_0.csv', delimiter=',', skiprows=1)
        #dataErrorFocMA = pd.read_csv('C:/Users/dkaklis/Desktop/TESTerrorPercFOC2_0MA.csv', delimiter=',', skiprows=1)
        dataErr = dataErrorFoc.values
        #dataErrMA = dataErrorFocMA.values
        errMean = []
        errMeanMA = []
        focMean = []
        i = 0
        while i < 2000:
            errMean.append(np.mean(dataErr[i:i + 20, 0]))
            #errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
            focMean.append(np.mean(dataErr[i:i + 20, 1]))


            i = i + 20

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



        plt.plot(np.linspace(0, 100, 100), errMean, '-', c='green', label='FOC Predictions')
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=stwMean,c='red', alpha=0.5)
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=draftMean, c='blue', alpha=0.5)

        #plt.plot(np.linspace(0, 100, 100), errMeanMA, '-', c='blue', label='FOC Predictions with moving avg')
        plt.plot(np.linspace(0, 100, 100), focMean, '-', c='red', label='Actual FOC')
        #plt.fill_between(np.linspace(0, 100, 100), errMean - mae, errMean + mae,color='gray', alpha=0.2)
        plt.ylabel('FOC')

        plt.legend()
        plt.grid()
        plt.show()

        x=0