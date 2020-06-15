import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
#from pyproj import Proj, transform
import pyearth as sp
import matplotlib
from sklearn.cluster import KMeans
from decimal import Decimal
import random
#from coordinates.converter import CoordinateConverter, WGS84, L_Est97
import pyodbc
import csv
import locale
#locale.setlocale(locale.LC_NUMERIC, "en_DK.UTF-8")
import datetime
#import cx_Oracle
from dateutil.rrule import rrule, DAILY, MINUTELY
from sympy.solvers import solve
from sympy import Symbol
from pathlib import Path
import itertools
from sympy import cos, sin , tan , exp , sqrt , E
from openpyxl import load_workbook
import glob, os
from pathlib import Path
from openpyxl.styles.colors import YELLOW
from openpyxl.styles import Font
from openpyxl.styles.borders import Border, Side
import shutil
from openpyxl.styles import PatternFill
from openpyxl.styles import Alignment
import matplotlib.pyplot as plt
#import seaborn as sns
from openpyxl.drawing.image import Image

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm

class BaseReportExtraction:

    def BuildPdfReport(self,company,vessel):



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
