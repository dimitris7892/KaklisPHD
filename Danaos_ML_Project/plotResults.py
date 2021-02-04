import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sqrt
import pandas as pd
import bokeh
import dask.dataframe as dd, geoviews as gv, cartopy.crs as crs
from colorcet import fire
import holoviews as hv
from datashader.utils import lnglat_to_meters
from holoviews.operation.datashader import datashade
from geoviews.tile_sources import EsriImagery
import holoviews.plotting.bokeh
import seaborn as sns
from bokeh.plotting import show
from beaufort_scale import beaufort_scale_kmh
from mpl_toolkits.basemap import Basemap

import matplotlib
#matplotlib.use('TkAgg')
import param, panel as pn
from colorcet import palette
hv.extension('bokeh', 'matplotlib')
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

        dataErrorFoc = pd.read_csv('./TESTerrorPercFOC1_0.csv', delimiter=',', skiprows=0)
        #dataErrorFocMA = pd.read_csv('C:/Users/dkaklis/Desktop/TESTerrorPercFOC2_0MA.csv', delimiter=',', skiprows=1)
        dataErr = dataErrorFoc.values
        #dataErrMA = dataErrorFocMA.values
        errMean = []
        errMeanMA = []
        focMean = []

        errMean020=[]
        focMean020=[]

        errMean2040 = []
        focMean2040 = []

        errMean4060=[]
        focMean4060 = []

        errMean6080 = []
        focMean6080 = []

        errMean80100 = []
        focMean80100 = []

        errMean100 = []
        focMean100 = []

        i = 8.75
        maxSpeed = np.max(dataErr[:, 3])
        sizesSpeed = []
        AvgactualFoc = []
        dt = dataErr
        speed = []
        AvgActualFoc=[]
        AvgPredFoc=[]
        speedSize = []
        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray = np.array([k for k in dataErr if float(k[3]) >= i and float(k[3]) <= i + 0.5])

            if speedArray.__len__() > 0:
                speedSize.append(len(speedArray))
                speed.append(i+0.25)
                AvgActualFoc.append(np.round( np.mean(speedArray[:, 1]),1))
                AvgPredFoc.append( np.round(np.mean(speedArray[:, 0]),1))
            i += 0.5

        d = {'Speed':speed,'AvgActualFoc': AvgPredFoc, 'AvgPredFoc': AvgActualFoc}
        df = pd.DataFrame(d)

        fig = plt.figure()
        #ax = fig.add_axes([0, 0, 1, 1])
        speeds = speed
        sizes = speedSize
        plt.bar(speeds, sizes)
        plt.plot(speeds, AvgActualFoc, '-', c='red', label='Actual FOC')
        plt.plot(speeds, AvgPredFoc, '-', c='green', label='Pred FOC')
        #plt.fill_between(np.linspace(0, 100, 100), errMean - mae, errMean + mae,color='gray', alpha=0.2)
        plt.ylabel('Count')
        plt.xlabel('Speed ranges')

        plt.legend()
        plt.grid()
        plt.show()

        df.to_csv('./data/DANAOS/EXPRESS ATHENS/AvgActualAvgPredSpeed.csv', index=False)
        #return
        i = 0
        size = 3500
        for i in range(0,len(dataErr)):
            #errMean.append(np.mean(dataErr[i:i + 20, 0]))
            errMean.append(dataErr[i,0])
            #errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
            focMean.append(dataErr[i, 1])

        for i in range(0, len(dataErr)):
                # errMean.append(np.mean(dataErr[i:i + 20, 0]))
            if dataErr[i,1] >10 and dataErr[i,1]<=20:
                    errMean020.append(dataErr[i, 0])
                    # errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
                    focMean020.append(dataErr[i, 1])

        for i in range(0, len(dataErr)):
            # errMean.append(np.mean(dataErr[i:i + 20, 0]))
            if dataErr[i, 1] > 20 and dataErr[i, 1] <= 30:
                errMean2040.append(dataErr[i, 0])
                # errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
                focMean2040.append(dataErr[i, 1])

        for i in range(0, len(dataErr)):
            # errMean.append(np.mean(dataErr[i:i + 20, 0]))
            if dataErr[i, 1] > 40 and dataErr[i, 1] <= 45:
                errMean4060.append(dataErr[i, 0])
                # errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
                focMean4060.append(dataErr[i, 1])

        for i in range(0, len(dataErr)):
            # errMean.append(np.mean(dataErr[i:i + 20, 0]))
            if dataErr[i, 1] > 60 and dataErr[i, 1] <= 80:
                errMean6080.append(dataErr[i, 0])
                # errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
                focMean6080.append(dataErr[i, 1])

        for i in range(0, len(dataErr)):
            # errMean.append(np.mean(dataErr[i:i + 20, 0]))
            if dataErr[i, 1] > 80 and dataErr[i, 1] <= 100:
                errMean80100.append(dataErr[i, 0])
                # errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
                focMean80100.append(dataErr[i, 1])

        for i in range(0, len(dataErr)):
            # errMean.append(np.mean(dataErr[i:i + 20, 0]))
            if dataErr[i, 1] >100:
                errMean100.append(dataErr[i, 0])
                # errMeanMA.append(np.mean(dataErrMA[i:i + 20, 0]))
                focMean100.append(dataErr[i, 1])

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

        fig, (ax1, ax2, ax3,) = plt.subplots( 3)
        #fig.suptitle('FOC ')
        ax1.plot(np.linspace(0, len(errMean020),len(errMean020)), errMean020, '-', c='green', label='FOC Predictions')
        ax1.plot(np.linspace(0, len(focMean020), len(focMean020)), focMean020, '-', c='red', label='Actual FOC')
        ax1.title.set_text("Actual FOC (10-20) (MT/day)")
        ax1.set_ylabel('FOC(lt/min)')
        ax1.set_xlabel('Observations')
        ax1.grid()

        ax2.title.set_text("Actual FOC (20-40)  (MT/day)")
        ax2.plot(np.linspace(0, len(errMean2040), len(errMean2040)), errMean2040, '-', c='green', label='FOC Predictions')
        ax2.plot(np.linspace(0, len(focMean2040), len(focMean2040)), focMean2040, '-', c='red', label='Actual FOC')
        ax2.set_ylabel('FOC(lt/min)')
        ax2.set_xlabel('Observations')
        ax2.grid()

        ax3.title.set_text("Actual FOC (40-60)  (MT/day)")
        ax3.plot(np.linspace(0, len(errMean4060), len(errMean4060)), errMean4060, '-', c='green',
                 label='FOC Predictions')
        ax3.plot(np.linspace(0, len(focMean4060), len(focMean4060)), focMean4060, '-', c='red', label='Actual FOC')
        ax3.grid()
        ax3.set_ylabel('FOC(lt/min)')
        ax3.set_xlabel('Observations')
        fig.tight_layout(pad=0.1)

        fig, (ax4,ax5,ax6) = plt.subplots(3)
        ax4.title.set_text("Actual FOC (60-80)  (MT/day)")
        ax4.plot(np.linspace(0, len(errMean6080), len(errMean6080)), errMean6080, '-', c='green',
                 label='FOC Predictions')
        ax4.plot(np.linspace(0, len(focMean6080), len(focMean6080)), focMean6080, '-', c='red', label='Actual FOC')
        ax4.set_ylabel('FOC(lt/min)')
        ax4.set_xlabel('Observations')
        ax4.grid()

        ax5.title.set_text("Actual FOC (80-100)  (MT/day)")
        ax5.plot(np.linspace(0, len(errMean80100), len(errMean80100)), errMean80100, '-', c='green',
                 label='FOC Predictions')
        ax5.plot(np.linspace(0, len(focMean80100), len(focMean80100)), focMean80100, '-', c='red', label='Actual FOC')
        ax5.set_ylabel('FOC(lt/min)')
        ax5.set_xlabel('Observations')
        ax5.legend()
        ax5.grid()

        ax6.title.set_text("Actual FOC (>100)  (MT/day)")
        ax6.plot(np.linspace(0, len(errMean100), len(errMean100)), errMean100, '-', c='green',
                 label='FOC Predictions')
        ax6.plot(np.linspace(0, len(focMean100), len(focMean100)), focMean100, '-', c='red', label='Actual FOC')
        ax6.set_ylabel('FOC(lt/min)')
        ax6.set_xlabel('Observations')
        ax6.legend()
        ax6.grid()

        fig.tight_layout(pad=0.1)
        #plt.plot(np.linspace(0, len(errMean),len(errMean)), errMean, '-', c='green', label='FOC Predictions')
        #plt.plot(np.linspace(0, len(focMean), len(focMean)), focMean, '-', c='red', label='Actual FOC')
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=stwMean,c='red', alpha=0.5)
        #plt.scatter(np.linspace(0, 100, 100), errMean, s=draftMean, c='blue', alpha=0.5)

        #plt.plot(np.linspace(0, 100, 100), errMeanMA, '-', c='blue', label='FOC Predictions with moving avg')
        #plt.plot(np.linspace(0, len(focMean), len(focMean)), focMean, '-', c='red', label='Actual FOC')
        #plt.fill_between(np.linspace(0, 100, 100), errMean - mae, errMean + mae,color='gray', alpha=0.2)
        #plt.ylabel('FOC(lt/min)')
        #plt.xlabel('Observations')

        #plt.legend()
        #plt.grid()
        plt.show()

        x=0

    def plotStatistics(self,data):

        x_1 = data.drop(["blFlags"], axis=1)  # .drop(drop_list1, axis=1)  # do not modify x, we will use it later
        x_1.head()
        f, ax = plt.subplots(figsize=(14, 14))
        sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
        plt.show()

        # data = data.drop(["wind_speed", "wind_dir","trim"], axis=1)
        x_train = data.drop(["blFlags", "focs", "tlgsFocs"], axis=1)

        foc = np.array(np.mean(data.values[:, 6:7], axis=1))
        y_train = pd.DataFrame({
            'FOC': foc,
        })

        clf_rf = RandomForestRegressor(random_state=43)
        clr_rf = clf_rf.fit(x_train, y_train)

        clf_rf_5 = RandomForestRegressor()
        clr_rf_5 = clf_rf_5.fit(x_train, y_train)
        importances = clr_rf_5.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(x_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest

        plt.figure(1, figsize=(14, 13))
        plt.title("Feature importances")
        plt.bar(range(x_train.shape[1]), importances[indices],
                color="g", yerr=std[indices], align="center")
        plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
        plt.xlim([-1, x_train.shape[1]])
        # plt.show()

    def PLotTrajectory(self,df,vesselName):

        class VesselTrajectory(param.Parameterized):
            alpha = param.Magnitude(default=0.9, doc="Map tile opacity")
            cmap = param.ObjectSelector('fire', objects=['fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc'])
            location = param.ObjectSelector(default='dropoff', objects=['dropoff', 'pickup'])

            def make_view(self, **kwargs):
                df.to_parquet('./data/'+vesselName+'CoorTest.parquet')
                topts = dict(width=1000, height=800, bgcolor='black', xaxis=None, yaxis=None, show_grid=True)
                tiles = EsriImagery.clone(crs=crs.GOOGLE_MERCATOR).options(**topts)
                dopts = dict(width=2000, height=800, x_sampling=0.5, y_sampling=0.5)

                route = dd.read_parquet('./data/'+vesselName+'CoorTest.parquet').persist()
                pts = hv.Points(route, ['x', 'y'])

                trips = datashade(pts, cmap=palette[self.cmap], **dopts)
                return tiles.options(alpha=self.alpha) * trips



        #df = pd.read_csv('./data/EXPRESS ATHENSCoorTest.csv')
        df.loc[:, 'x'], df.loc[:, 'y'] = lnglat_to_meters(df.Longitude, df.Latitude)
        #df.to_csv('./data/EXPRESS ATHENSCoor1.csv')
        '''df.to_parquet('./data/EXPRESS ATHENSCoorTest.parquet')
        topts = dict(width=700, height=600, bgcolor='black', xaxis=None, yaxis=None, show_grid=False)
        tiles = EsriImagery.clone(crs=crs.GOOGLE_MERCATOR).options(**topts)
        dopts = dict(width=2000, height=600, x_sampling=0.5, y_sampling=0.5)

        taxi = dd.read_parquet('./data/EXPRESS ATHENSCoorTest.parquet').persist()
        pts = hv.Points(taxi, ['x', 'y'])
        trips = datashade(pts, cmap=fire, **dopts)
        layout = tiles * trips'''

        explorer = VesselTrajectory(name=vesselName)
        pn.Row(explorer.param, explorer.make_view).show()

        #show(hv.render(layout))

    def PLotDists(self,data):

        #data[:, 15] = ((data[:, 15]) / 1000) * 1440

        foc1 = data[27000:86000, :]#.astype(float)
        
        foc1_912 = np.array([k for k in foc1 if k[3]>=9 and k[3]<=12])[:,5]
        foc1_1215 = np.array([k for k in foc1 if k[3] > 12 and k[3] <= 15])[:, 5]
        foc1_1518 = np.array([k for k in foc1 if k[3] > 15 and k[3] <= 18])[:, 5]
        foc1_1821 = np.array([k for k in foc1 if k[3] > 18 and k[3] <= 21])[:, 5]
        
        foc2 = data[86000:145115, :]#.astype(float)

        foc2_912 = np.array([k for k in foc2 if k[3] >= 9 and k[3] <= 12])[:, 5]
        foc2_1215 = np.array([k for k in foc2 if k[3] > 12 and k[3] <= 15])[:, 5]
        foc2_1518 = np.array([k for k in foc2 if k[3] > 15 and k[3] <= 18])[:, 5]
        foc2_1821 = np.array([k for k in foc2 if k[3] > 18 and k[3] <= 21])[:, 5]

        i=9
        maxSpeed = 20
        speed = []
        focs1 = []
        focs2 = []
        speed1 = []
        speed2 = []
        sizes1 = []
        sizes2 = []
        draft1=[]
        draft2 = []
        AvgPredFoc = []
        speedSize1 = []
        speedSize2 = []



        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray1 = np.array([k for k in foc1 if float(k[3]) >= i and float(k[3]) <= i + 1])
            speedArray2 = np.array([k for k in foc2 if float(k[3]) >= i and float(k[3]) <= i + 1])


            if speedArray1.__len__() > 0:
                #sizes1.append(len(speedArray1))
                sizes1.append(np.mean(speedArray1[:,2]))
                draft1.append(np.mean(speedArray1[:,0]))
                speed1.append(i)
                speedSize1.append(len(speedArray1))

                focs1.append(np.mean(speedArray1[:, 5]))
            if speedArray2.__len__() > 0:
                #sizes2.append(len(speedArray2))
                sizes2.append(np.mean(speedArray2[:, 2]))
                draft2.append(np.mean(speedArray2[:, 0]))
                speed2.append(i)
                speedSize2.append(len(speedArray2))

                focs2.append(np.mean(speedArray2[:, 5]))
            i +=1


        fig = plt.figure()
        # ax = fig.add_axes([0, 0, 1, 1])

        #sizes = speedSize
        plt.bar(speed1, sizes1,color='red',width=0.25,label='windSpeed1',)
        plt.bar(np.array(speed2) + 0.25, sizes2,color='green',width=0.25,label='windSpeed2',)
        '''foc1Arr = np.column_stack((speed1[0], focs1[0]))
        foc2Arr = np.column_stack((speed1[0], focs1[0]))

        foc1Arr =   foc1Arr[foc1Arr[:, 0].argsort()]
        foc2Arr = foc2Arr[foc2Arr[:, 0].argsort()]'''
        plt.scatter(speed1, focs1, s=np.array(draft1)*10, c="red", alpha=0.4, linewidth=4,label='dradt1',marker='o')
        plt.scatter(speed2, focs2, s=np.array(draft2)*10, c="green", alpha=0.4, linewidth=4,label='dradt2',marker='o')

        plt.plot(speed1, focs1, '-', c='red', label='FOC1 (2019-10-18) - (2019-12-11)',)
        plt.plot(speed2, focs2, '-', c='green', label='FOC2 (2019-12-12) - (2020-02-05)',)
        # plt.fill_between(np.linspace(0, 100, 100), errMean - mae, errMean + mae,color='gray', alpha=0.2)
        plt.ylabel('FOC')
        plt.xlabel('Speed ranges')

        plt.legend()
        plt.grid()
        #plt.show()

        foc12 = [(itm, '(2019-10-18) - (2019-12-11)') for itm in foc1[:,5]]
        foc23 = [(itm, '(2019-12-12) - (2020-02-05)') for itm in foc2[:,5]]

        joinedFoc = foc12 + foc23

        df = pd.DataFrame(data=joinedFoc,
                          columns=['foc', 'period'])
        # df.Zip = df.Zip.astype(str).str.zfill(5)

        #plt.title('FOC distributions')

        sns.displot(df, x="foc", hue='period')

        #################################################
        ws12 = [(itm, '(2019-10-18) - (2019-12-11)') for itm in foc1[:, 2]]
        ws23 = [(itm, '(2019-12-12) - (2020-02-05)') for itm in foc2[:, 2]]

        joinedFoc = ws12 + ws23

        df = pd.DataFrame(data=joinedFoc,
                          columns=['WindSpeed', 'period'])
        # df.Zip = df.Zip.astype(str).str.zfill(5)

        # plt.title('FOC distributions')

        sns.displot(df, x="WindSpeed", hue='period')


        plt.show()
