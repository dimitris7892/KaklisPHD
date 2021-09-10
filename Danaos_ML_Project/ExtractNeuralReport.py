import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
#from pyproj import Proj, transform
import pyearth as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
from math import sqrt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import bokeh
import scipy
from svglib.svglib import svg2rlg
from fastapi import APIRouter
import os

from reportlab.graphics import renderPDF

import io
import seaborn as sns
from sklearn.linear_model import LinearRegression
import json
from pandas import json_normalize
import requests
from scipy import spatial
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd, geoviews as gv, cartopy.crs as crs
from colorcet import fire
import holoviews as hv
from datashader.utils import lnglat_to_meters
from holoviews.operation.datashader import datashade
from geoviews.tile_sources import EsriImagery
import holoviews.plotting.bokeh
import seaborn as sns
from bokeh.plotting import show, figure
from beaufort_scale import beaufort_scale_kmh
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib
import param, panel as pn
from colorcet import palette
from sklearn.cluster import KMeans , DBSCAN
from sklearn.decomposition import PCA
from wand.image import Image as WImage
import matplotlib.cm as cm
from IPython.display import Image
import itertools
import plotly.graph_objects as go
import math
import plotly
from scipy import spatial,random
import dataframe_image as dfi
import colorsys
import csv
from sklearn.decomposition import PCA
import glob , path
import glob, os
from pathlib import Path
#from openpyxl.styles.colors import YELLOW
from openpyxl.styles import Font
from openpyxl.styles.borders import Border, Side
import shutil
from openpyxl.styles import PatternFill
from openpyxl.styles import Alignment
import matplotlib.pyplot as plt
#import seaborn as sns
from PIL import Image
from openpyxl.drawing.image import Image
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
hv.extension('bokeh', 'matplotlib')



class BaseReportExtraction:

    def BuildPdfReportForNeural(self,company,vessel):



        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(17.5, 9.5)

        dataSetN = []
        dataSetP = []
        dataSetError = []
        dataSetTrue = []
        for infile in sorted(glob.glob('./data/' + company + '/' + vessel + '/NeuralEvaluationData/TRAINerrorPercFOC7_0.csv')):
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=1)
            dataSetTrue.append(data.values[:, 1])  # subsets=[]
            dataSetError.append(data.values[:, 2])  # subsets=[]

            print(str(infile))  # for i in range(1,5):
        # dataSetError = np.concatenate(dataSetError)
        dataSetTrue = np.concatenate(dataSetTrue)
        dataSetError = np.concatenate(dataSetError)

        dtTPreds = []

        for i in range(0, len(dataSetError)):
            dtTPreds.append(float(dataSetError[i].split('[')[2].split(']')[0]))

        dataSetError = np.array(dtTPreds)

        dataSetN = np.array(np.append(dataSetTrue.reshape(-1, 1), np.asmatrix([dataSetError]).T, axis=1)).astype(float)

        for infile in sorted(glob.glob('./data/' + company + '/' + vessel + '/NeuralEvaluationData/TRAINerrorPercFOCPavlos0.csv')):
            data = pd.read_csv(infile, sep=',', decimal='.', skiprows=1)
            dataSetP.append(data.values)
            print(str(infile))
        dataSetP = np.concatenate(dataSetP)
        maxSpeed = int(np.ceil(np.max(dataSetP[:, 1])))
        minSpeed = int(np.ceil(np.min(dataSetP[:, 1])))

        i = 3
        rangesN = []
        percsN = []
        sizeN = []
        while i <= maxSpeed:

            lenRange = len(np.array([k for k in dataSetN if k[0] >= i and k[0] <= i + 5]))
            if lenRange > 0:
                rangeSTWErrN = np.mean(np.array([k for k in dataSetN if k[0] >= i and k[0] <= i + 5])[:, 1])
            percRange = rangeSTWErrN if lenRange > 0 else 0
            rangesN.append(i)
            percsN.append(percRange)
            sizeN.append(lenRange)
            i = i + 5

        i = 3
        rangesP = []
        percsP = []
        sizeP = []

        while i <= maxSpeed:

            lenRange = len(np.array([k for k in dataSetP if k[1] >= i and k[1] <= i + 5]))
            if lenRange > 0:
                rangeSTWErrP = np.mean(np.array([k for k in dataSetP if k[1] >= i and k[1] <= i + 5])[:, 2])
            percRange = rangeSTWErrP if lenRange > 0 else 0
            rangesP.append(i)
            percsP.append(percRange)
            sizeP.append(lenRange)
            i = i + 5
        # indSubsets = []

        # unseensY = []
        sizeN = [k / 10 for k in sizeN]
        axes = plt.gca()
        # axes.set_xlim([minSpeed, maxSpeed])
        # axes.set_ylim([0, 40])
        plt.step(rangesN, percsN, color='blue', label='Neural')
        plt.plot(rangesN, percsN, 'o--', color='grey', alpha=0.4)
        plt.scatter(rangesN, percsN, s=sizeN, alpha=0.4)

        plt.step(rangesP, percsP, color='orange', label='Interpolation')
        plt.plot(rangesP, percsP, 'o--', color='grey', alpha=0.4)
        # plt.xticks(np.array(np.linspace(1, maxSpeed, maxSpeed)))
        # plt.yticks(np.array(np.linspace(1, 40, 10)))
        plt.xlabel("FOC")
        plt.ylabel("Mean Percentage Error")
        plt.title("Mean Percentage Error on train data (80 * 10^3) obs.")
        # plt.scatter(ranges, percs, s=size, c="blue", alpha=0.4, linewidth=4)    #k = n * i + 10
        plt.legend()
        # plt.show()

        fig.savefig('./Figures/' + company + '_' + vessel + '_NeuralConv.png', dpi=96)
        plt.cla()
        data = pd.read_csv('./data/' + company + '/' + vessel + '/NeuralEvaluationData/errorEpochCLusters.csv')
        clusters = data.values[:,0]
        sizeTrDt = data.values[:, 1]
        clScores = data.values[:, 3]
        minEpochs = data.values[:, 4]

        plt.plot(clusters, sizeTrDt, color='orange', label='Size of Cluster')
        plt.plot(clusters, clScores, 'purple', label='Validation Error')
        plt.xlabel("clusters")
        plt.ylabel("validation MSE")
        plt.scatter(clusters, sizeTrDt, s=clScores, c="blue", alpha=0.4, linewidth=4)
        plt.legend()
        # plt.show()



        fig.savefig('./Figures/' + company + '_' + vessel + '_EpochErrorClusters.png', dpi=96)

        c = canvas.Canvas('ex.pdf')
        c.drawImage('./Figures/' + company + '_' + vessel + '_NeuralConv.png', 60, 60, 10 * cm, 15 * cm)
        c.drawImage('./Figures/' + company + '_' + vessel + '_EpochErrorClusters.png', 20, 20, 20 * cm, 15 * cm)

        c.save()

    def extractDataModerna(self,company,vessel):

        dataSet = []
        path = './data/' + company + '/' + vessel + '/'
        for infile in sorted(glob.glob(path + '*.xlsx')):
            data = pd.read_excel(infile, skiprows=1, index_col=0, header=None)
            dfTranposed = data.T
            dfTranposed = dfTranposed.drop(['REPORT'], axis=1)
            dataSet.append(dfTranposed)
            print(str(infile))

        dataSet = pd.concat(dataSet)
        print(dataSet.shape)
        dataSet.head(10)

        dfTranposed = dataSet

        ####FEATURE ENGINEERING

        ##draft
        draft = dfTranposed['Draft (Fwd / Aft) (m)'].values
        # print(draft)
        midDraft = np.array([(float(k.split("/")[0]) + float(k.split("/")[1])) / 2 for k in draft])
        midDraft
        dfTranposed['midDraft'] = midDraft

        ##foc
        totalFoc = dfTranposed['LSFO consumption'].values + dfTranposed['LSMGO consumption'].values
        totalFoc = (totalFoc * 24) / dfTranposed['Steaming Time (Hrs.)']
        dfTranposed.insert(7, 'totalFoc', totalFoc)

        weightedtotalFoc = np.average(totalFoc, weights=dfTranposed['Steaming Time (Hrs.)'])
        # print("Weighted avg foc: " +str(weightedtotalFoc))
        ##speed
        dist = np.array([int(k) for k in dfTranposed['Distance'].values])
        time = np.array([float(k) for k in dfTranposed['Steaming Time (Hrs.)'].values])
        correctedSpeed = np.round(dist / time, 2)
        for i in range(0, len(correctedSpeed)):
            if correctedSpeed[i] > 30:
                correctedSpeed[i] = dfTranposed['STW'].values[i]
        # print(correctedSpeed)
        dfTranposed.insert(8, 'correctedSpeed', correctedSpeed)

        # dfTranposed['correctedSpeed'] = correctedSpeed
        print("Shape: " + str(dfTranposed.shape))

        dfTranposed = dfTranposed.reset_index(drop=True)
        df_styled = dfTranposed.head(10).style.background_gradient()
        dfi.export(df_styled, './data/'+company+'/'+vessel+'/data.png')
        # dfi
        return dfTranposed

    def meanFocSpeedGraphs(self,company,vessel, dfTranposed):

        dataValues = dfTranposed.values
        i = np.min(dfTranposed['STW'])
        maxSpeed = np.max(dfTranposed['STW'])
        sizesSpeed = []
        speed = []
        avgActualFoc = []

        minActualFoc = []
        maxActualFoc = []
        stdActualFoc = []
        speed = []
        avgActualFocPerMile = []
        minActualFocPerMile = []
        maxActualFocPerMile = []
        stdActualFocPerMile = []
        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray = np.array([k for k in dataValues if float(k[8]) >= i - 0.25 and float(k[8]) <= i + 0.25])

            if speedArray.__len__() > 0:
                sizesSpeed.append(len(speedArray))
                speed.append(i)
                avgActualFoc.append(np.mean(speedArray[:, 7]))
                minActualFoc.append(np.min(speedArray[:, 7]))
                maxActualFoc.append(np.max(speedArray[:, 7]))
                stdActualFoc.append(np.std(speedArray[:, 7]))

                # avgActualFocPerMile.append(np.mean(speedArray[:, 7]))
                # minActualFocPerMile.append(np.min(speedArray[:, 7]))
                # maxActualFocPerMile.append(np.max(speedArray[:, 7]))
                # stdActualFocPerMile.append(np.std(speedArray[:, 7]))
                # if i >=16.25 and i <= 16.75 :
                # print(np.mean(speedArray1[:,5]))
            i += 0.5

        plt.scatter(speed, avgActualFoc, s=np.array(sizesSpeed) * 20, c='green', zorder=20)
        plt.grid()
        plt.plot(speed, avgActualFoc, label='Mean Foc / speed / # of obs', lw=3, zorder=20, color='red')
        plt.legend()
        # plt.ylim(5,50)
        plt.xlabel('Speed')
        plt.ylabel('Foc')
        plt.savefig('./data/' + company + '/' + vessel + '/Figures/focSpeed.svg')
        #plt.show()
        plt.clf()

        #############################
        dataValues = dfTranposed.values
        i = np.min(dfTranposed['Wind Speed (BF)'])
        maxSpeed = np.max(dfTranposed['Wind Speed (BF)'])
        sizesSpeed = []
        speed = []
        avgActualFoc = []

        minActualFoc = []
        maxActualFoc = []
        stdActualFoc = []
        windSpeed = []
        avgActualFocPerMile = []
        minActualFocPerMile = []
        maxActualFocPerMile = []
        stdActualFocPerMile = []
        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray = np.array([k for k in dataValues if float(k[16]) >= i - 1 and float(k[16]) <= i + 1])

            if speedArray.__len__() > 0:
                sizesSpeed.append(len(speedArray))
                speed.append(i)
                windSpeed.append(np.mean(speedArray[:, 16]))
                avgActualFoc.append(np.mean(speedArray[:, 7]))
                minActualFoc.append(np.min(speedArray[:, 7]))
                maxActualFoc.append(np.max(speedArray[:, 7]))
                stdActualFoc.append(np.std(speedArray[:, 7]))

                # avgActualFocPerMile.append(np.mean(speedArray[:, 7]))
                # minActualFocPerMile.append(np.min(speedArray[:, 7]))
                # maxActualFocPerMile.append(np.max(speedArray[:, 7]))
                # stdActualFocPerMile.append(np.std(speedArray[:, 7]))
                # if i >=16.25 and i <= 16.75 :
                # print(np.mean(speedArray1[:,5]))
            i += 1

        plt.scatter(speed, avgActualFoc, s=np.array(sizesSpeed) * 10, c='green', zorder=20)
        plt.grid()
        plt.plot(speed, avgActualFoc, label='Mean Foc / Wind speed / # of obs', lw=2, zorder=20, color='red')
        plt.legend()
        plt.xlabel('Wind speed')
        plt.ylabel('FOC')
        plt.savefig('./data/' + company + '/' + vessel + '/Figures/focWS.svg')

        #####################
        plt.clf()
        i = np.min(dfTranposed['Swell Height'])
        maxSpeed = np.max(dfTranposed['Swell Height'])
        sizesSpeed = []
        speed = []
        avgActualFoc = []

        minActualFoc = []
        maxActualFoc = []
        stdActualFoc = []
        windSpeed = []
        avgActualFocPerMile = []
        minActualFocPerMile = []
        maxActualFocPerMile = []
        stdActualFocPerMile = []
        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray = np.array([k for k in dataValues if float(k[22]) >= i - 0.5 and float(k[22]) <= i + 0.5])

            if speedArray.__len__() > 0:
                sizesSpeed.append(len(speedArray))
                speed.append(i)
                windSpeed.append(np.mean(speedArray[:, 22]))
                avgActualFoc.append(np.mean(speedArray[:, 7]))
                minActualFoc.append(np.min(speedArray[:, 7]))
                maxActualFoc.append(np.max(speedArray[:, 7]))
                stdActualFoc.append(np.std(speedArray[:, 7]))

                # avgActualFocPerMile.append(np.mean(speedArray[:, 7]))
                # minActualFocPerMile.append(np.min(speedArray[:, 7]))
                # maxActualFocPerMile.append(np.max(speedArray[:, 7]))
                # stdActualFocPerMile.append(np.std(speedArray[:, 7]))
                # if i >=16.25 and i <= 16.75 :
                # print(np.mean(speedArray1[:,5]))
            i += 0.5

        plt.scatter(speed, avgActualFoc, s=np.array(sizesSpeed) * 20, c='blue', zorder=20)
        plt.grid()
        plt.plot(speed, avgActualFoc, label='Mean Foc / Swell Height / # of obs', lw=3, zorder=20, color='red')
        plt.legend()
        plt.xlabel('Swell Height')
        plt.ylabel('FOC')
        plt.tight_layout()
        plt.savefig('./data/' + company + '/' + vessel + '/Figures/focSWH.svg')

        plt.clf()
        wh = list(dfTranposed['Wave Height (m)'])
        print(wh.index('`'))
        wh = np.array([float(k) for k in wh if k != '`'])
        foc = np.array(dfTranposed['totalFoc'])
        foc = np.delete(foc, [22])
        whFoc = np.append(wh.reshape(-1, 1), foc.reshape(-1, 1), axis=1)
        print(whFoc[0])
        i = np.min(wh)
        maxSpeed = np.max(wh)
        sizesSpeed = []
        speed = []
        avgActualFoc = []

        minActualFoc = []
        maxActualFoc = []
        stdActualFoc = []
        windSpeed = []
        avgActualFocPerMile = []
        minActualFocPerMile = []
        maxActualFocPerMile = []
        stdActualFocPerMile = []
        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray = np.array([k for k in whFoc if float(k[0]) >= i - 0.5 and float(k[0]) <= i + 0.5])

            if speedArray.__len__() > 0:
                sizesSpeed.append(len(speedArray))
                speed.append(i)
                windSpeed.append(np.mean(speedArray[:, 0]))
                avgActualFoc.append(np.mean(speedArray[:, 1]))
                minActualFoc.append(np.min(speedArray[:, 1]))
                maxActualFoc.append(np.max(speedArray[:, 1]))
                stdActualFoc.append(np.std(speedArray[:, 1]))

                # avgActualFocPerMile.append(np.mean(speedArray[:, 7]))
                # minActualFocPerMile.append(np.min(speedArray[:, 7]))
                # maxActualFocPerMile.append(np.max(speedArray[:, 7]))
                # stdActualFocPerMile.append(np.std(speedArray[:, 7]))
                # if i >=16.25 and i <= 16.75 :
                # print(np.mean(speedArray1[:,5]))
            i += 0.5

        plt.scatter(speed, avgActualFoc, s=np.array(sizesSpeed) * 20, c='blue', zorder=20)
        plt.grid()
        plt.plot(speed, avgActualFoc, label='Mean Foc / Wave Height(m) / # of obs', lw=3, zorder=20, color='red')
        plt.legend()
        plt.xlabel('Wave Height (m)')
        plt.ylabel('Foc')
        plt.tight_layout()
        plt.savefig('./data/' + company + '/' + vessel + '/Figures/focWH.svg')


        ####################
        plt.clf()
        i = np.min(dfTranposed['Current Speed (kts.)'])
        maxSpeed = np.max(dfTranposed['Current Speed (kts.)'])
        sizesSpeed = []
        speed = []
        avgActualFoc = []

        minActualFoc = []
        maxActualFoc = []
        stdActualFoc = []
        windSpeed = []
        avgActualFocPerMile = []
        minActualFocPerMile = []
        maxActualFocPerMile = []
        stdActualFocPerMile = []
        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray = np.array([k for k in dataValues if float(k[18]) >= i - 0.5 and float(k[18]) <= i + 0.5])

            if speedArray.__len__() > 0:
                sizesSpeed.append(len(speedArray))
                speed.append(i)
                windSpeed.append(np.mean(speedArray[:, 18]))
                avgActualFoc.append(np.mean(speedArray[:, 7]))
                minActualFoc.append(np.min(speedArray[:, 7]))
                maxActualFoc.append(np.max(speedArray[:, 7]))
                stdActualFoc.append(np.std(speedArray[:, 7]))

                # avgActualFocPerMile.append(np.mean(speedArray[:, 7]))
                # minActualFocPerMile.append(np.min(speedArray[:, 7]))
                # maxActualFocPerMile.append(np.max(speedArray[:, 7]))
                # stdActualFocPerMile.append(np.std(speedArray[:, 7]))
                # if i >=16.25 and i <= 16.75 :
                # print(np.mean(speedArray1[:,5]))
            i += 0.5

        plt.scatter(speed, avgActualFoc, s=np.array(sizesSpeed) * 20, c='blue', zorder=20)
        plt.grid()
        plt.plot(speed, avgActualFoc, label='Mean Foc / Current Speed / # of obs', lw=3, zorder=20, color='red')
        plt.legend()
        plt.xlabel('Current Speed')
        plt.ylabel('Foc')
        plt.tight_layout()
        plt.savefig('./data/' + company + '/' + vessel + '/Figures/focCS.svg')

    def buildPdfReportModerna(self,company,vessel):

        dfTranposed = self.extractDataModerna(company,vessel)
        self.meanFocSpeedGraphs(company,vessel,dfTranposed)

        plt.clf()
        plt.plot(dfTranposed['correctedSpeed'].values, label='correctedSpeed')
        plt.plot(dfTranposed['STW'].values, label='stw original')
        plt.grid()
        plt.ylim(3, 20)
        plt.xlabel('# of obs')
        plt.legend()
        plt.ylabel('speed')
        plt.savefig('./data/'+company+'/'+vessel+'/Figures/corrSpeed_stw.svg')


        sns_displot_stw = sns.displot(dfTranposed['STW'])
        sns_displot_corrSpeed = sns.displot(dfTranposed['correctedSpeed'])

        sns_displot_midDraft = sns.displot(dfTranposed['midDraft'])
        sns_displot_totalFoc = sns.displot(dfTranposed['totalFoc'])
        sns_displot_ws = sns.displot(dfTranposed['Wind Speed (BF)'])
        sns_displot_cs = sns.displot(dfTranposed['Current Speed (kts.)'])
        # sns.displot(dfTranposed['Wave Height (m)'])
        sns_displot_swh = sns.displot(dfTranposed['Swell Height'])

        sns_displot_stw.savefig('./data/'+company+'/'+vessel+'/Figures/displot1.png')
        sns_displot_corrSpeed.savefig('./data/' + company + '/' + vessel + '/Figures/displot2.png')
        sns_displot_midDraft.savefig('./data/' + company + '/' + vessel + '/Figures/displot3.png')
        sns_displot_totalFoc.savefig('./data/' + company + '/' + vessel + '/Figures/displot4.png')
        sns_displot_ws.savefig('./data/' + company + '/' + vessel + '/Figures/displot5.png')
        sns_displot_cs.savefig('./data/' + company + '/' + vessel + '/Figures/displot6.png')
        sns_displot_swh.savefig('./data/' + company + '/' + vessel + '/Figures/displot7.png')



        c = canvas.Canvas('./data/'+company+'/'+vessel+'/'+vessel+'_.pdf')

        #for k in range(1,8):
        c.setFont('Helvetica-Bold', 24)
        c.drawCentredString(300, 600, "Spetses Spirit Report")
        c.showPage()

        c.setFont('Helvetica-Bold', 20)
        c.drawCentredString(700, 800, "Dataset Snapshots")
        c.setPageSize((1500, 1000))
        c.drawImage('./data/' + company + '/' + vessel + '/Figures/snapshotData_1.png', 70, 300)
        c.showPage()

        c.setFont('Helvetica-Bold', 20)
        c.setPageSize((650,900))
        c.drawCentredString(300, 800, "Plot Distributions")
        c.drawImage('./data/' + company + '/' + vessel + '/Figures/displot1.png', 30, 500, 8 * cm, 8 * cm)
        c.drawImage('./data/' + company + '/' + vessel + '/Figures/displot2.png', 350, 500, 8 * cm, 8 * cm)

        c.drawImage('./data/' + company + '/' + vessel + '/Figures/displot3.png', 30, 200, 8 * cm, 8 * cm)
        c.drawImage('./data/' + company + '/' + vessel + '/Figures/displot4.png', 350, 200, 8 * cm, 8 * cm)
        c.showPage()



        c.drawImage('./data/' + company + '/' + vessel + '/Figures/displot5.png', 30, 500, 8 * cm, 8 * cm)
        c.drawImage('./data/' + company + '/' + vessel + '/Figures/displot6.png', 350, 500, 8 * cm, 8 * cm)

        c.drawImage('./data/' + company + '/' + vessel + '/Figures/displot7.png', 150, 200, 8 * cm, 8 * cm)

        c.showPage()

        c.setFont('Helvetica-Bold', 20)
        c.drawCentredString(300, 800, "Speed corrected with Steaming Time")

        drawing = svg2rlg('./data/' + company + '/' + vessel + '/Figures/corrSpeed_stw.svg')
        drawing.scale(0.7, 0.7)
        renderPDF.draw(drawing, c, 100, 400)
        c.showPage()

        c.setFont('Helvetica-Bold', 20)
        c.drawCentredString(300, 800, "Plot Main FOC Graphs")
        drawing = svg2rlg('./data/' + company + '/' + vessel + '/Figures/focSpeed.svg')
        drawing.scale(0.7, 0.7)
        renderPDF.draw(drawing, c, 100, 400)
        c.showPage()
        drawing = svg2rlg('./data/' + company + '/' + vessel + '/Figures/focWS.svg')
        drawing.scale(0.7, 0.7)
        renderPDF.draw(drawing, c, 100, 400)
        c.showPage()
        drawing = svg2rlg('./data/' + company + '/' + vessel + '/Figures/focSWH.svg')
        drawing.scale(0.7, 0.7)
        renderPDF.draw(drawing, c, 100, 400)
        c.showPage()
        drawing = svg2rlg('./data/' + company + '/' + vessel + '/Figures/focWH.svg')
        drawing.scale(0.7,0.7)
        renderPDF.draw(drawing, c, 100, 400)
        c.showPage()

        drawing = svg2rlg('./data/' + company + '/' + vessel + '/Figures/focCS.svg')
        drawing.scale(0.7, 0.7)
        renderPDF.draw(drawing,c,100, 400)

        c.showPage()

        c.save()


#br = BaseReportExtraction()
#br.buildPdfReportModerna('MODERNA','SPETSES SPIRIT')